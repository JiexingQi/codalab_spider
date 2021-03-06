/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <thrift/compiler/ast/t_function.h>
#include <thrift/compiler/ast/t_type.h>

namespace apache {
namespace thrift {
namespace compiler {

class t_program;

/**
 * A service consists of a set of functions.
 */
class t_service : public t_type {
 public:
  explicit t_service(
      t_program* program, std::string name, const t_service* extends = nullptr)
      : t_type(program, std::move(name)), extends_(extends) {}

  const t_service* extends() const { return extends_; }

  const std::vector<const t_function*>& functions() const {
    return functions_raw_;
  }
  void set_functions(std::vector<std::unique_ptr<t_function>> functions);
  void add_function(std::unique_ptr<t_function> func);

  type get_type_value() const override { return type::t_service; }
  bool is_service() const override { return true; }

  std::string get_full_name() const override {
    return make_full_name("service");
  }

 private:
  std::vector<std::unique_ptr<t_function>> functions_;
  std::vector<const t_function*> functions_raw_;

  const t_service* extends_ = nullptr;

  // TODO(afuller): Remove everything below this comment. It is only provided
  // for backwards compatibility.
  std::vector<t_function*> old_functions_raw_;

 public:
  const std::vector<t_function*>& get_functions() const {
    return old_functions_raw_;
  }

  const t_service* get_extends() const { return extends_; }
  void set_extends(const t_service* extends) { extends_ = extends; }

  bool is_interaction() const;
  bool is_serial_interaction() const;
};

} // namespace compiler
} // namespace thrift
} // namespace apache
