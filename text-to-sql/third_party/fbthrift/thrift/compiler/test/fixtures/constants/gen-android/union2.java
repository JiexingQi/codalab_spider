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

@SuppressWarnings({ "unused", "serial", "unchecked" })
public class union2 extends TUnion<union2> {
  private static final TStruct STRUCT_DESC = new TStruct("union2");
  private static final TField I_FIELD_DESC = new TField("i", TType.I32, (short)1);
  private static final TField D_FIELD_DESC = new TField("d", TType.DOUBLE, (short)2);
  private static final TField S_FIELD_DESC = new TField("s", TType.STRUCT, (short)3);
  private static final TField U_FIELD_DESC = new TField("u", TType.STRUCT, (short)4);

  public static final int I = 1;
  public static final int D = 2;
  public static final int S = 3;
  public static final int U = 4;

  public static final Map<Integer, FieldMetaData> metaDataMap = new HashMap<>();

  public union2() {
    super();
  }

  public union2(int setField, Object __value) {
    super(setField, __value);
  }

  public union2(union2 other) {
    super(other);
  }

  public union2 deepCopy() {
    return new union2(this);
  }

  public static union2 i(Integer __value) {
    union2 x = new union2();
    x.setI(__value);
    return x;
  }

  public static union2 d(Double __value) {
    union2 x = new union2();
    x.setD(__value);
    return x;
  }

  public static union2 s(struct1 __value) {
    union2 x = new union2();
    x.setS(__value);
    return x;
  }

  public static union2 u(union1 __value) {
    union2 x = new union2();
    x.setU(__value);
    return x;
  }


  @Override
  protected Object readValue(TProtocol iprot, TField __field) throws TException {
    switch (__field.id) {
      case I:
        if (__field.type == I_FIELD_DESC.type) {
          Integer i;
          i = iprot.readI32();
          return i;
        }
        break;
      case D:
        if (__field.type == D_FIELD_DESC.type) {
          Double d;
          d = iprot.readDouble();
          return d;
        }
        break;
      case S:
        if (__field.type == S_FIELD_DESC.type) {
          struct1 s;
          s = struct1.deserialize(iprot);
          return s;
        }
        break;
      case U:
        if (__field.type == U_FIELD_DESC.type) {
          union1 u;
          u = new union1();
          u.read(iprot);
          return u;
        }
        break;
    }
    TProtocolUtil.skip(iprot, __field.type);
    return null;
  }

  @Override
  protected void writeValue(TProtocol oprot, short setField, Object __value) throws TException {
    switch (setField) {
      case I:
        Integer i = (Integer)getFieldValue();
        oprot.writeI32(i);
        return;
      case D:
        Double d = (Double)getFieldValue();
        oprot.writeDouble(d);
        return;
      case S:
        struct1 s = (struct1)getFieldValue();
        s.write(oprot);
        return;
      case U:
        union1 u = (union1)getFieldValue();
        u.write(oprot);
        return;
      default:
        throw new IllegalStateException("Cannot write union with unknown field " + setField);
    }
  }

  @Override
  protected TField getFieldDesc(int setField) {
    switch (setField) {
      case I:
        return I_FIELD_DESC;
      case D:
        return D_FIELD_DESC;
      case S:
        return S_FIELD_DESC;
      case U:
        return U_FIELD_DESC;
      default:
        throw new IllegalArgumentException("Unknown field id " + setField);
    }
  }

  @Override
  protected TStruct getStructDesc() {
    return STRUCT_DESC;
  }

  @Override
  protected Map<Integer, FieldMetaData> getMetaDataMap() { return metaDataMap; }

  private Object __getValue(int expectedFieldId) {
    if (getSetField() == expectedFieldId) {
      return getFieldValue();
    } else {
      throw new RuntimeException("Cannot get field '" + getFieldDesc(expectedFieldId).name + "' because union is currently set to " + getFieldDesc(getSetField()).name);
    }
  }

  private void __setValue(int fieldId, Object __value) {
    if (__value == null) throw new NullPointerException();
    setField_ = fieldId;
    value_ = __value;
  }

  public Integer getI() {
    return (Integer) __getValue(I);
  }

  public void setI(Integer __value) {
    __setValue(I, __value);
  }

  public Double getD() {
    return (Double) __getValue(D);
  }

  public void setD(Double __value) {
    __setValue(D, __value);
  }

  public struct1 getS() {
    return (struct1) __getValue(S);
  }

  public void setS(struct1 __value) {
    __setValue(S, __value);
  }

  public union1 getU() {
    return (union1) __getValue(U);
  }

  public void setU(union1 __value) {
    __setValue(U, __value);
  }

  public boolean equals(Object other) {
    if (other instanceof union2) {
      return equals((union2)other);
    } else {
      return false;
    }
  }

  public boolean equals(union2 other) {
    return equalsNobinaryImpl(other);
  }


  @Override
  public int hashCode() {
    return Arrays.deepHashCode(new Object[] {getSetField(), getFieldValue()});
  }

}
