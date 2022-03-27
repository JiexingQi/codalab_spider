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

#include <string>

#include <thrift/compiler/ast/t_named.h>

namespace apache {
namespace thrift {
namespace compiler {

/**
 * class t_enum_value
 *
 * A constant. These are used inside of enum definitions. Constants are just
 * symbol identifiers that may or may not have an explicit value associated
 * with them.
 *
 */
class t_enum_value : public t_named {
 public:
  t_enum_value() = default;

  /**
   * t_enum_value setters
   */
  void set_value(int32_t value) {
    value_ = value;
    has_value_ = true;
  }

  void set_implicit_value(int32_t value) {
    value_ = value;
    has_value_ = false;
  }

  /**
   * t_enum_value getters
   */
  int32_t get_value() const { return value_; }

  bool has_value() { return has_value_; }

 private:
  int32_t value_{0};
  bool has_value_{false};
};

} // namespace compiler
} // namespace thrift
} // namespace apache
