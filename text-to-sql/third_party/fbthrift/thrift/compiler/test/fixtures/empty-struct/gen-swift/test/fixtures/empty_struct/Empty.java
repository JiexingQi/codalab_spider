/**
 * Autogenerated by Thrift
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */

package test.fixtures.empty_struct;

import com.facebook.swift.codec.*;
import com.facebook.swift.codec.ThriftField.Requiredness;
import com.facebook.swift.codec.ThriftField.Recursiveness;
import com.google.common.collect.*;
import java.util.*;
import org.apache.thrift.*;
import org.apache.thrift.async.*;
import org.apache.thrift.meta_data.*;
import org.apache.thrift.server.*;
import org.apache.thrift.transport.*;
import org.apache.thrift.protocol.*;
import org.apache.thrift.meta_data.FieldValueMetaData;
import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.base.MoreObjects.ToStringHelper;

@SwiftGenerated
@com.facebook.swift.codec.ThriftStruct(value="Empty", builder=Empty.Builder.class)
public final class Empty {

    @ThriftConstructor
    public Empty(
    ) {
    }
    
    
    
    public static class Builder {
    
    
    public Builder() { }
        public Builder(Empty other) {
        }
    
        @ThriftConstructor
        public Empty build() {
            Empty result = new Empty (
            );
            return result;
        }
    }
    
    public static final Map<String, Integer> NAMES_TO_IDS = new HashMap();
    public static final Map<Integer, TField> FIELD_METADATA = new HashMap<>();
    private static final TStruct STRUCT_DESC = new TStruct("Empty");
static {
    }
    @java.lang.Override
    public String toString() {
        ToStringHelper helper = toStringHelper(this);
        return helper.toString();
    }
    
    @java.lang.Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
    
        Empty other = (Empty)o;
    
        return
            true;
    }
    
    @java.lang.Override
    public int hashCode() {
        return Arrays.deepHashCode(new Object[] {
        });
    }
    
    
    public static Empty read0(TProtocol oprot) throws TException {
      TField __field;
      oprot.readStructBegin(Empty.NAMES_TO_IDS, Empty.FIELD_METADATA);
      Empty.Builder builder = new Empty.Builder();
      while (true) {
        __field = oprot.readFieldBegin();
        if (__field.type == TType.STOP) { break; }
        switch (__field.id) {
        default:
          TProtocolUtil.skip(oprot, __field.type);
          break;
        }
        oprot.readFieldEnd();
      }
      oprot.readStructEnd();
      return builder.build();
    }
    
    public void write0(TProtocol oprot) throws TException {
      oprot.writeStructBegin(STRUCT_DESC);
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }
    
    private static class _EmptyLazy {
        private static final Empty _DEFAULT = new Empty.Builder().build();
    }
    
    public static Empty defaultInstance() {
        return  _EmptyLazy._DEFAULT;
    }
}
