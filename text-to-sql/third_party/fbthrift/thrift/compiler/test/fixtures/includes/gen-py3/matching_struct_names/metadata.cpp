/**
 * Autogenerated by Thrift
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */

#include "src/gen-py3/matching_struct_names/metadata.h"

#include <thrift/lib/py3/metadata.h>

namespace cpp2 {
::apache::thrift::metadata::ThriftMetadata matching_struct_names_getThriftModuleMetadata() {
  ::apache::thrift::metadata::ThriftMetadata metadata;
  ::apache::thrift::metadata::ThriftServiceContext serviceContext;
  ::apache::thrift::detail::md::StructMetadata<MyStruct>::gen(metadata);
  ::apache::thrift::detail::md::StructMetadata<Combo>::gen(metadata);
  return metadata;
}
} // namespace cpp2
