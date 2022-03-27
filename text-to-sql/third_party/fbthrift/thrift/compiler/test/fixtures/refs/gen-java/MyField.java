/**
 * Autogenerated by Thrift
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.Collections;
import java.util.BitSet;
import java.util.Arrays;
import com.facebook.thrift.*;
import com.facebook.thrift.annotations.*;
import com.facebook.thrift.async.*;
import com.facebook.thrift.meta_data.*;
import com.facebook.thrift.server.*;
import com.facebook.thrift.transport.*;
import com.facebook.thrift.protocol.*;

@SuppressWarnings({ "unused", "serial" })
public class MyField implements TBase, java.io.Serializable, Cloneable, Comparable<MyField> {
  private static final TStruct STRUCT_DESC = new TStruct("MyField");
  private static final TField OPT_VALUE_FIELD_DESC = new TField("opt_value", TType.I64, (short)1);
  private static final TField VALUE_FIELD_DESC = new TField("value", TType.I64, (short)2);
  private static final TField REQ_VALUE_FIELD_DESC = new TField("req_value", TType.I64, (short)3);
  private static final TField OPT_ENUM_VALUE_FIELD_DESC = new TField("opt_enum_value", TType.I32, (short)4);
  private static final TField ENUM_VALUE_FIELD_DESC = new TField("enum_value", TType.I32, (short)5);
  private static final TField REQ_ENUM_VALUE_FIELD_DESC = new TField("req_enum_value", TType.I32, (short)6);

  public long opt_value;
  public long value;
  public long req_value;
  /**
   * 
   * @see MyEnum
   */
  public MyEnum opt_enum_value;
  /**
   * 
   * @see MyEnum
   */
  public MyEnum enum_value;
  /**
   * 
   * @see MyEnum
   */
  public MyEnum req_enum_value;
  public static final int OPT_VALUE = 1;
  public static final int VALUE = 2;
  public static final int REQ_VALUE = 3;
  public static final int OPT_ENUM_VALUE = 4;
  public static final int ENUM_VALUE = 5;
  public static final int REQ_ENUM_VALUE = 6;

  // isset id assignments
  private static final int __OPT_VALUE_ISSET_ID = 0;
  private static final int __VALUE_ISSET_ID = 1;
  private static final int __REQ_VALUE_ISSET_ID = 2;
  private BitSet __isset_bit_vector = new BitSet(3);

  public static final Map<Integer, FieldMetaData> metaDataMap;

  static {
    Map<Integer, FieldMetaData> tmpMetaDataMap = new HashMap<Integer, FieldMetaData>();
    tmpMetaDataMap.put(OPT_VALUE, new FieldMetaData("opt_value", TFieldRequirementType.OPTIONAL, 
        new FieldValueMetaData(TType.I64)));
    tmpMetaDataMap.put(VALUE, new FieldMetaData("value", TFieldRequirementType.DEFAULT, 
        new FieldValueMetaData(TType.I64)));
    tmpMetaDataMap.put(REQ_VALUE, new FieldMetaData("req_value", TFieldRequirementType.REQUIRED, 
        new FieldValueMetaData(TType.I64)));
    tmpMetaDataMap.put(OPT_ENUM_VALUE, new FieldMetaData("opt_enum_value", TFieldRequirementType.OPTIONAL, 
        new FieldValueMetaData(TType.I32)));
    tmpMetaDataMap.put(ENUM_VALUE, new FieldMetaData("enum_value", TFieldRequirementType.DEFAULT, 
        new FieldValueMetaData(TType.I32)));
    tmpMetaDataMap.put(REQ_ENUM_VALUE, new FieldMetaData("req_enum_value", TFieldRequirementType.REQUIRED, 
        new FieldValueMetaData(TType.I32)));
    metaDataMap = Collections.unmodifiableMap(tmpMetaDataMap);
  }

  static {
    FieldMetaData.addStructMetaDataMap(MyField.class, metaDataMap);
  }

  public MyField() {
  }

  public MyField(
      long req_value,
      MyEnum req_enum_value) {
    this();
    this.req_value = req_value;
    setReq_valueIsSet(true);
    this.req_enum_value = req_enum_value;
  }

  public MyField(
      long value,
      long req_value,
      MyEnum enum_value,
      MyEnum req_enum_value) {
    this();
    this.value = value;
    setValueIsSet(true);
    this.req_value = req_value;
    setReq_valueIsSet(true);
    this.enum_value = enum_value;
    this.req_enum_value = req_enum_value;
  }

  public MyField(
      long opt_value,
      long value,
      long req_value,
      MyEnum opt_enum_value,
      MyEnum enum_value,
      MyEnum req_enum_value) {
    this();
    this.opt_value = opt_value;
    setOpt_valueIsSet(true);
    this.value = value;
    setValueIsSet(true);
    this.req_value = req_value;
    setReq_valueIsSet(true);
    this.opt_enum_value = opt_enum_value;
    this.enum_value = enum_value;
    this.req_enum_value = req_enum_value;
  }

  public static class Builder {
    private long opt_value;
    private long value;
    private long req_value;
    private MyEnum opt_enum_value;
    private MyEnum enum_value;
    private MyEnum req_enum_value;

    BitSet __optional_isset = new BitSet(3);

    public Builder() {
    }

    public Builder setOpt_value(final long opt_value) {
      this.opt_value = opt_value;
      __optional_isset.set(__OPT_VALUE_ISSET_ID, true);
      return this;
    }

    public Builder setValue(final long value) {
      this.value = value;
      __optional_isset.set(__VALUE_ISSET_ID, true);
      return this;
    }

    public Builder setReq_value(final long req_value) {
      this.req_value = req_value;
      __optional_isset.set(__REQ_VALUE_ISSET_ID, true);
      return this;
    }

    public Builder setOpt_enum_value(final MyEnum opt_enum_value) {
      this.opt_enum_value = opt_enum_value;
      return this;
    }

    public Builder setEnum_value(final MyEnum enum_value) {
      this.enum_value = enum_value;
      return this;
    }

    public Builder setReq_enum_value(final MyEnum req_enum_value) {
      this.req_enum_value = req_enum_value;
      return this;
    }

    public MyField build() {
      MyField result = new MyField();
      if (__optional_isset.get(__OPT_VALUE_ISSET_ID)) {
        result.setOpt_value(this.opt_value);
      }
      if (__optional_isset.get(__VALUE_ISSET_ID)) {
        result.setValue(this.value);
      }
      if (__optional_isset.get(__REQ_VALUE_ISSET_ID)) {
        result.setReq_value(this.req_value);
      }
      result.setOpt_enum_value(this.opt_enum_value);
      result.setEnum_value(this.enum_value);
      result.setReq_enum_value(this.req_enum_value);
      return result;
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Performs a deep copy on <i>other</i>.
   */
  public MyField(MyField other) {
    __isset_bit_vector.clear();
    __isset_bit_vector.or(other.__isset_bit_vector);
    this.opt_value = TBaseHelper.deepCopy(other.opt_value);
    this.value = TBaseHelper.deepCopy(other.value);
    this.req_value = TBaseHelper.deepCopy(other.req_value);
    if (other.isSetOpt_enum_value()) {
      this.opt_enum_value = TBaseHelper.deepCopy(other.opt_enum_value);
    }
    if (other.isSetEnum_value()) {
      this.enum_value = TBaseHelper.deepCopy(other.enum_value);
    }
    if (other.isSetReq_enum_value()) {
      this.req_enum_value = TBaseHelper.deepCopy(other.req_enum_value);
    }
  }

  public MyField deepCopy() {
    return new MyField(this);
  }

  public long getOpt_value() {
    return this.opt_value;
  }

  public MyField setOpt_value(long opt_value) {
    this.opt_value = opt_value;
    setOpt_valueIsSet(true);
    return this;
  }

  public void unsetOpt_value() {
    __isset_bit_vector.clear(__OPT_VALUE_ISSET_ID);
  }

  // Returns true if field opt_value is set (has been assigned a value) and false otherwise
  public boolean isSetOpt_value() {
    return __isset_bit_vector.get(__OPT_VALUE_ISSET_ID);
  }

  public void setOpt_valueIsSet(boolean __value) {
    __isset_bit_vector.set(__OPT_VALUE_ISSET_ID, __value);
  }

  public long getValue() {
    return this.value;
  }

  public MyField setValue(long value) {
    this.value = value;
    setValueIsSet(true);
    return this;
  }

  public void unsetValue() {
    __isset_bit_vector.clear(__VALUE_ISSET_ID);
  }

  // Returns true if field value is set (has been assigned a value) and false otherwise
  public boolean isSetValue() {
    return __isset_bit_vector.get(__VALUE_ISSET_ID);
  }

  public void setValueIsSet(boolean __value) {
    __isset_bit_vector.set(__VALUE_ISSET_ID, __value);
  }

  public long getReq_value() {
    return this.req_value;
  }

  public MyField setReq_value(long req_value) {
    this.req_value = req_value;
    setReq_valueIsSet(true);
    return this;
  }

  public void unsetReq_value() {
    __isset_bit_vector.clear(__REQ_VALUE_ISSET_ID);
  }

  // Returns true if field req_value is set (has been assigned a value) and false otherwise
  public boolean isSetReq_value() {
    return __isset_bit_vector.get(__REQ_VALUE_ISSET_ID);
  }

  public void setReq_valueIsSet(boolean __value) {
    __isset_bit_vector.set(__REQ_VALUE_ISSET_ID, __value);
  }

  /**
   * 
   * @see MyEnum
   */
  public MyEnum getOpt_enum_value() {
    return this.opt_enum_value;
  }

  /**
   * 
   * @see MyEnum
   */
  public MyField setOpt_enum_value(MyEnum opt_enum_value) {
    this.opt_enum_value = opt_enum_value;
    return this;
  }

  public void unsetOpt_enum_value() {
    this.opt_enum_value = null;
  }

  // Returns true if field opt_enum_value is set (has been assigned a value) and false otherwise
  public boolean isSetOpt_enum_value() {
    return this.opt_enum_value != null;
  }

  public void setOpt_enum_valueIsSet(boolean __value) {
    if (!__value) {
      this.opt_enum_value = null;
    }
  }

  /**
   * 
   * @see MyEnum
   */
  public MyEnum getEnum_value() {
    return this.enum_value;
  }

  /**
   * 
   * @see MyEnum
   */
  public MyField setEnum_value(MyEnum enum_value) {
    this.enum_value = enum_value;
    return this;
  }

  public void unsetEnum_value() {
    this.enum_value = null;
  }

  // Returns true if field enum_value is set (has been assigned a value) and false otherwise
  public boolean isSetEnum_value() {
    return this.enum_value != null;
  }

  public void setEnum_valueIsSet(boolean __value) {
    if (!__value) {
      this.enum_value = null;
    }
  }

  /**
   * 
   * @see MyEnum
   */
  public MyEnum getReq_enum_value() {
    return this.req_enum_value;
  }

  /**
   * 
   * @see MyEnum
   */
  public MyField setReq_enum_value(MyEnum req_enum_value) {
    this.req_enum_value = req_enum_value;
    return this;
  }

  public void unsetReq_enum_value() {
    this.req_enum_value = null;
  }

  // Returns true if field req_enum_value is set (has been assigned a value) and false otherwise
  public boolean isSetReq_enum_value() {
    return this.req_enum_value != null;
  }

  public void setReq_enum_valueIsSet(boolean __value) {
    if (!__value) {
      this.req_enum_value = null;
    }
  }

  public void setFieldValue(int fieldID, Object __value) {
    switch (fieldID) {
    case OPT_VALUE:
      if (__value == null) {
        unsetOpt_value();
      } else {
        setOpt_value((Long)__value);
      }
      break;

    case VALUE:
      if (__value == null) {
        unsetValue();
      } else {
        setValue((Long)__value);
      }
      break;

    case REQ_VALUE:
      if (__value == null) {
        unsetReq_value();
      } else {
        setReq_value((Long)__value);
      }
      break;

    case OPT_ENUM_VALUE:
      if (__value == null) {
        unsetOpt_enum_value();
      } else {
        setOpt_enum_value((MyEnum)__value);
      }
      break;

    case ENUM_VALUE:
      if (__value == null) {
        unsetEnum_value();
      } else {
        setEnum_value((MyEnum)__value);
      }
      break;

    case REQ_ENUM_VALUE:
      if (__value == null) {
        unsetReq_enum_value();
      } else {
        setReq_enum_value((MyEnum)__value);
      }
      break;

    default:
      throw new IllegalArgumentException("Field " + fieldID + " doesn't exist!");
    }
  }

  public Object getFieldValue(int fieldID) {
    switch (fieldID) {
    case OPT_VALUE:
      return new Long(getOpt_value());

    case VALUE:
      return new Long(getValue());

    case REQ_VALUE:
      return new Long(getReq_value());

    case OPT_ENUM_VALUE:
      return getOpt_enum_value();

    case ENUM_VALUE:
      return getEnum_value();

    case REQ_ENUM_VALUE:
      return getReq_enum_value();

    default:
      throw new IllegalArgumentException("Field " + fieldID + " doesn't exist!");
    }
  }

  @Override
  public boolean equals(Object _that) {
    if (_that == null)
      return false;
    if (this == _that)
      return true;
    if (!(_that instanceof MyField))
      return false;
    MyField that = (MyField)_that;

    if (!TBaseHelper.equalsNobinary(this.isSetOpt_value(), that.isSetOpt_value(), this.opt_value, that.opt_value)) { return false; }

    if (!TBaseHelper.equalsNobinary(this.value, that.value)) { return false; }

    if (!TBaseHelper.equalsNobinary(this.req_value, that.req_value)) { return false; }

    if (!TBaseHelper.equalsNobinary(this.isSetOpt_enum_value(), that.isSetOpt_enum_value(), this.opt_enum_value, that.opt_enum_value)) { return false; }

    if (!TBaseHelper.equalsNobinary(this.isSetEnum_value(), that.isSetEnum_value(), this.enum_value, that.enum_value)) { return false; }

    if (!TBaseHelper.equalsNobinary(this.isSetReq_enum_value(), that.isSetReq_enum_value(), this.req_enum_value, that.req_enum_value)) { return false; }

    return true;
  }

  @Override
  public int hashCode() {
    return Arrays.deepHashCode(new Object[] {opt_value, value, req_value, opt_enum_value, enum_value, req_enum_value});
  }

  @Override
  public int compareTo(MyField other) {
    if (other == null) {
      // See java.lang.Comparable docs
      throw new NullPointerException();
    }

    if (other == this) {
      return 0;
    }
    int lastComparison = 0;

    lastComparison = Boolean.valueOf(isSetOpt_value()).compareTo(other.isSetOpt_value());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(opt_value, other.opt_value);
    if (lastComparison != 0) { 
      return lastComparison;
    }
    lastComparison = Boolean.valueOf(isSetValue()).compareTo(other.isSetValue());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(value, other.value);
    if (lastComparison != 0) { 
      return lastComparison;
    }
    lastComparison = Boolean.valueOf(isSetReq_value()).compareTo(other.isSetReq_value());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(req_value, other.req_value);
    if (lastComparison != 0) { 
      return lastComparison;
    }
    lastComparison = Boolean.valueOf(isSetOpt_enum_value()).compareTo(other.isSetOpt_enum_value());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(opt_enum_value, other.opt_enum_value);
    if (lastComparison != 0) { 
      return lastComparison;
    }
    lastComparison = Boolean.valueOf(isSetEnum_value()).compareTo(other.isSetEnum_value());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(enum_value, other.enum_value);
    if (lastComparison != 0) { 
      return lastComparison;
    }
    lastComparison = Boolean.valueOf(isSetReq_enum_value()).compareTo(other.isSetReq_enum_value());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(req_enum_value, other.req_enum_value);
    if (lastComparison != 0) { 
      return lastComparison;
    }
    return 0;
  }

  public void read(TProtocol iprot) throws TException {
    TField __field;
    iprot.readStructBegin(metaDataMap);
    while (true)
    {
      __field = iprot.readFieldBegin();
      if (__field.type == TType.STOP) { 
        break;
      }
      switch (__field.id)
      {
        case OPT_VALUE:
          if (__field.type == TType.I64) {
            this.opt_value = iprot.readI64();
            setOpt_valueIsSet(true);
          } else { 
            TProtocolUtil.skip(iprot, __field.type);
          }
          break;
        case VALUE:
          if (__field.type == TType.I64) {
            this.value = iprot.readI64();
            setValueIsSet(true);
          } else { 
            TProtocolUtil.skip(iprot, __field.type);
          }
          break;
        case REQ_VALUE:
          if (__field.type == TType.I64) {
            this.req_value = iprot.readI64();
            setReq_valueIsSet(true);
          } else { 
            TProtocolUtil.skip(iprot, __field.type);
          }
          break;
        case OPT_ENUM_VALUE:
          if (__field.type == TType.I32) {
            this.opt_enum_value = MyEnum.findByValue(iprot.readI32());
          } else { 
            TProtocolUtil.skip(iprot, __field.type);
          }
          break;
        case ENUM_VALUE:
          if (__field.type == TType.I32) {
            this.enum_value = MyEnum.findByValue(iprot.readI32());
          } else { 
            TProtocolUtil.skip(iprot, __field.type);
          }
          break;
        case REQ_ENUM_VALUE:
          if (__field.type == TType.I32) {
            this.req_enum_value = MyEnum.findByValue(iprot.readI32());
          } else { 
            TProtocolUtil.skip(iprot, __field.type);
          }
          break;
        default:
          TProtocolUtil.skip(iprot, __field.type);
          break;
      }
      iprot.readFieldEnd();
    }
    iprot.readStructEnd();


    // check for required fields of primitive type, which can't be checked in the validate method
    if (!isSetReq_value()) {
      throw new TProtocolException("Required field 'req_value' was not found in serialized data! Struct: " + toString());
    }
    validate();
  }

  public void write(TProtocol oprot) throws TException {
    validate();

    oprot.writeStructBegin(STRUCT_DESC);
    if (isSetOpt_value()) {
      oprot.writeFieldBegin(OPT_VALUE_FIELD_DESC);
      oprot.writeI64(this.opt_value);
      oprot.writeFieldEnd();
    }
    oprot.writeFieldBegin(VALUE_FIELD_DESC);
    oprot.writeI64(this.value);
    oprot.writeFieldEnd();
    oprot.writeFieldBegin(REQ_VALUE_FIELD_DESC);
    oprot.writeI64(this.req_value);
    oprot.writeFieldEnd();
    if (this.opt_enum_value != null) {
      if (isSetOpt_enum_value()) {
        oprot.writeFieldBegin(OPT_ENUM_VALUE_FIELD_DESC);
        oprot.writeI32(this.opt_enum_value == null ? 0 : this.opt_enum_value.getValue());
        oprot.writeFieldEnd();
      }
    }
    if (this.enum_value != null) {
      oprot.writeFieldBegin(ENUM_VALUE_FIELD_DESC);
      oprot.writeI32(this.enum_value == null ? 0 : this.enum_value.getValue());
      oprot.writeFieldEnd();
    }
    if (this.req_enum_value != null) {
      oprot.writeFieldBegin(REQ_ENUM_VALUE_FIELD_DESC);
      oprot.writeI32(this.req_enum_value == null ? 0 : this.req_enum_value.getValue());
      oprot.writeFieldEnd();
    }
    oprot.writeFieldStop();
    oprot.writeStructEnd();
  }

  @Override
  public String toString() {
    return toString(1, true);
  }

  @Override
  public String toString(int indent, boolean prettyPrint) {
    String indentStr = prettyPrint ? TBaseHelper.getIndentedString(indent) : "";
    String newLine = prettyPrint ? "\n" : "";
    String space = prettyPrint ? " " : "";
    StringBuilder sb = new StringBuilder("MyField");
    sb.append(space);
    sb.append("(");
    sb.append(newLine);
    boolean first = true;

    if (isSetOpt_value())
    {
      sb.append(indentStr);
      sb.append("opt_value");
      sb.append(space);
      sb.append(":").append(space);
      sb.append(TBaseHelper.toString(this.getOpt_value(), indent + 1, prettyPrint));
      first = false;
    }
    if (!first) sb.append("," + newLine);
    sb.append(indentStr);
    sb.append("value");
    sb.append(space);
    sb.append(":").append(space);
    sb.append(TBaseHelper.toString(this.getValue(), indent + 1, prettyPrint));
    first = false;
    if (!first) sb.append("," + newLine);
    sb.append(indentStr);
    sb.append("req_value");
    sb.append(space);
    sb.append(":").append(space);
    sb.append(TBaseHelper.toString(this.getReq_value(), indent + 1, prettyPrint));
    first = false;
    if (isSetOpt_enum_value())
    {
      if (!first) sb.append("," + newLine);
      sb.append(indentStr);
      sb.append("opt_enum_value");
      sb.append(space);
      sb.append(":").append(space);
      if (this.getOpt_enum_value() == null) {
        sb.append("null");
      } else {
        String opt_enum_value_name = this.getOpt_enum_value() == null ? "null" : this.getOpt_enum_value().name();
        if (opt_enum_value_name != null) {
          sb.append(opt_enum_value_name);
          sb.append(" (");
        }
        sb.append(this.getOpt_enum_value());
        if (opt_enum_value_name != null) {
          sb.append(")");
        }
      }
      first = false;
    }
    if (!first) sb.append("," + newLine);
    sb.append(indentStr);
    sb.append("enum_value");
    sb.append(space);
    sb.append(":").append(space);
    if (this.getEnum_value() == null) {
      sb.append("null");
    } else {
      String enum_value_name = this.getEnum_value() == null ? "null" : this.getEnum_value().name();
      if (enum_value_name != null) {
        sb.append(enum_value_name);
        sb.append(" (");
      }
      sb.append(this.getEnum_value());
      if (enum_value_name != null) {
        sb.append(")");
      }
    }
    first = false;
    if (!first) sb.append("," + newLine);
    sb.append(indentStr);
    sb.append("req_enum_value");
    sb.append(space);
    sb.append(":").append(space);
    if (this.getReq_enum_value() == null) {
      sb.append("null");
    } else {
      String req_enum_value_name = this.getReq_enum_value() == null ? "null" : this.getReq_enum_value().name();
      if (req_enum_value_name != null) {
        sb.append(req_enum_value_name);
        sb.append(" (");
      }
      sb.append(this.getReq_enum_value());
      if (req_enum_value_name != null) {
        sb.append(")");
      }
    }
    first = false;
    sb.append(newLine + TBaseHelper.reduceIndent(indentStr));
    sb.append(")");
    return sb.toString();
  }

  public void validate() throws TException {
    // check for required fields
    // alas, we cannot check 'req_value' because it's a primitive and you chose the non-beans generator.
    if (req_enum_value == null) {
      throw new TProtocolException(TProtocolException.MISSING_REQUIRED_FIELD, "Required field 'req_enum_value' was not present! Struct: " + toString());
    }
  }

}
