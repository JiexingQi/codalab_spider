/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fizz/experimental/crypto/BatchSignature.h>
#include <fizz/experimental/protocol/BatchSignatureTypes.h>
#include <fizz/server/AsyncSelfCert.h>
#include <folly/futures/SharedPromise.h>

namespace fizz {

template <typename Hash>
class Batcher {
 public:
  /**
   * The data structure used to fulfill the promise.
   */
  struct SignedTree {
    SignedTree(
        std::shared_ptr<const BatchSignatureMerkleTree<Hash>> tree,
        std::shared_ptr<const folly::fbstring> signature)
        : tree_(std::move(tree)), signature_(std::move(signature)) {}
    std::shared_ptr<const BatchSignatureMerkleTree<Hash>> tree_;
    std::shared_ptr<const folly::fbstring> signature_;
  };

  /**
   * The data structure kept by the consumers.
   */
  struct BatchResult {
    BatchResult(size_t index, folly::SemiFuture<SignedTree>&& signedTreeFuture)
        : index_(index), future_(std::move(signedTreeFuture)) {}

    size_t index_;
    folly::SemiFuture<SignedTree> future_;
  };

  /**
   * The data structure kept by the producer to store the data for each epoch.
   */
  struct EpochData {
    EpochData(size_t numMsgThreshold)
        : tree_(std::make_shared<BatchSignatureMerkleTree<Hash>>(
              numMsgThreshold)) {}
    std::shared_ptr<BatchSignatureMerkleTree<Hash>> tree_;
    folly::SharedPromise<SignedTree> promise_;
  };

  /**
   * Constructor a batcher used for collecting messages from TLS handshakes.
   *
   * @param numMsgThreshold threshold that will trigger batch signing.
   * @param signer          the SelfCert used for base signature signing.
   * @param context         the context for signature signing.
   */
  Batcher(
      size_t numMsgThreshold,
      std::shared_ptr<SelfCert> signer,
      CertificateVerifyContext context)
      : numMsgThreshold_(numMsgThreshold), signer_(signer), context_(context) {
    auto result = signer_->getSigSchemes();
    // select the first matched batch scheme as batcher's scheme because each
    // batcher instance can at most support one batch signature scheme
    for (const auto& scheme : result) {
      auto batchScheme = BatchSignatureSchemes<Hash>::getFromBaseScheme(scheme);
      if (batchScheme) {
        batchScheme_ = *batchScheme;
        baseScheme_ = scheme;
      }
    }
  }
  virtual ~Batcher() = default;

  /**
   * Add a new message into the underlying Merkle Tree and get a future copy of
   * a finalized tree.
   *
   * The future tree will be fulfilled when the underlying Merkle Tree is ready
   * to be signed.
   */
  virtual BatchResult addMessageAndSign(folly::ByteRange msg) = 0;

  /**
   * Get the supported batch signature scheme.
   */
  SignatureScheme getBatchScheme() const {
    return batchScheme_;
  }

  std::shared_ptr<const SelfCert> getSigner() const {
    return signer_;
  }

 protected:
  folly::Future<SignedTree> signMerkleTree(
      std::shared_ptr<BatchSignatureMerkleTree<Hash>>&& tree) {
    tree->finalizeTree();
    auto toBeSigned =
        BatchSignature::encodeToBeSigned(tree->getRootValue(), batchScheme_);
    auto asyncSigner = dynamic_cast<const AsyncSelfCert*>(signer_.get());
    folly::Future<folly::Optional<Buf>> signatureFut = folly::none;
    if (asyncSigner) {
      signatureFut = asyncSigner->signFuture(
          baseScheme_, context_, toBeSigned->coalesce());
    } else {
      signatureFut = folly::makeFuture(
          signer_->sign(baseScheme_, context_, toBeSigned->coalesce()));
    }
    return std::move(signatureFut)
        .thenValue(
            [treeCapture = std::move(tree)](folly::Optional<Buf>&& signature) {
              if (!signature.has_value()) {
                throw std::runtime_error(
                    "Base signature cannot be generated by the signer.");
              }
              return SignedTree(
                  std::move(treeCapture),
                  std::make_shared<const folly::fbstring>(
                      std::move(signature).value()->moveToFbString()));
            });
  }

  size_t numMsgThreshold_;
  std::shared_ptr<const SelfCert> signer_;
  CertificateVerifyContext context_;
  SignatureScheme batchScheme_;
  SignatureScheme baseScheme_;
};

/**
 * A batcher to store and manage the Merkle Tree globally for multiple threads.
 * The underlying Merkle Tree will be shared by all threads.
 */
template <typename Hash = Sha256>
class SynchronizedBatcher : public Batcher<Hash> {
 public:
  SynchronizedBatcher(
      size_t numMsgThreshold,
      std::shared_ptr<SelfCert> signer,
      CertificateVerifyContext context)
      : Batcher<Hash>(numMsgThreshold, signer, context),
        epoch_(numMsgThreshold) {}

  typename Batcher<Hash>::BatchResult addMessageAndSign(
      folly::ByteRange msg) override {
    auto epoch = epoch_.wlock();
    auto index = epoch->tree_->appendTranscript(msg);
    if (!index.hasValue()) {
      throw std::runtime_error(
          "Cannot append more TLS transcripts into the Merkle Tree");
    };
    VLOG(5) << "Adding message to batch. batcher=" << this
            << ", added_message_index= " << index.value()
            << ", threshold=" << this->numMsgThreshold_;
    typename Batcher<Hash>::BatchResult result(
        index.value(), epoch->promise_.getSemiFuture());
    if (epoch->tree_->countMessages() >= this->numMsgThreshold_) {
      // We have reached the threshold, and will complete this epoch.
      // Perform tree finalization and signatures outside of the lock, as this
      // is expensive.
      auto oldTree = std::move(epoch->tree_);
      auto oldPromise = std::move(epoch->promise_);
      epoch->tree_ = std::make_shared<BatchSignatureMerkleTree<Hash>>(
          this->numMsgThreshold_);
      epoch->promise_ =
          folly::SharedPromise<typename Batcher<Hash>::SignedTree>();
      epoch.unlock();
      VLOG(5) << "Batcher reached message threshold. batcher=" << this
              << ", threshold=" << this->numMsgThreshold_;
      this->signMerkleTree(std::move(oldTree))
          .thenTry([promiseCapture = std::move(oldPromise)](
                       folly::Try<typename Batcher<Hash>::SignedTree>&&
                           signedTree) mutable {
            if (signedTree.hasValue()) {
              promiseCapture.setValue(std::move(signedTree).value());
            } else {
              promiseCapture.setException(std::move(signedTree).exception());
            }
          });
    }
    return result;
  }

 private:
  folly::Synchronized<typename Batcher<Hash>::EpochData> epoch_;
};

/**
 * A batcher to store and manage the Merkle Tree for each thread.
 * Each thread will have a Merkle Tree.
 */
template <typename Hash = Sha256>
class ThreadLocalBatcher : public Batcher<Hash> {
 public:
  ThreadLocalBatcher(
      size_t numMsgThreshold,
      std::shared_ptr<SelfCert> signer,
      CertificateVerifyContext context)
      : Batcher<Hash>(numMsgThreshold, signer, context), epoch_([=]() {
          return new typename Batcher<Hash>::EpochData(numMsgThreshold);
        }) {}

  typename Batcher<Hash>::BatchResult addMessageAndSign(
      folly::ByteRange msg) override {
    auto index = epoch_->tree_->appendTranscript(msg);
    if (!index.hasValue()) {
      throw std::runtime_error(
          "Cannot append more TLS transcripts into the Merkle Tree");
    };
    VLOG(5) << "Adding message to batch. batcher=" << this
            << ", added_message_index= " << index.value()
            << ", threshold=" << this->numMsgThreshold_;
    typename Batcher<Hash>::BatchResult result(
        index.value(), epoch_->promise_.getSemiFuture());
    if (epoch_->tree_->countMessages() >= this->numMsgThreshold_) {
      // We have reached the threshold, and will complete this epoch.
      auto oldTree = std::move(epoch_->tree_);
      auto oldPromise = std::move(epoch_->promise_);
      epoch_.reset();
      VLOG(5) << "Batcher reached message threshold. batcher=" << this
              << ", threshold=" << this->numMsgThreshold_;
      this->signMerkleTree(std::move(oldTree))
          .thenTry([promiseCapture = std::move(oldPromise)](
                       folly::Try<typename Batcher<Hash>::SignedTree>&&
                           signedTree) mutable {
            if (signedTree.hasValue()) {
              promiseCapture.setValue(std::move(signedTree).value());
            } else {
              promiseCapture.setException(std::move(signedTree).exception());
            }
          });
    }
    return result;
  }

 private:
  folly::ThreadLocal<typename Batcher<Hash>::EpochData> epoch_;
};

} // namespace fizz
