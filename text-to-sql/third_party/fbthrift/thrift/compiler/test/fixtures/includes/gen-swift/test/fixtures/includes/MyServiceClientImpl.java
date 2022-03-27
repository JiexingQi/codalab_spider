/**
 * Autogenerated by Thrift
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */

package test.fixtures.includes;

import com.facebook.nifty.client.RequestChannel;
import com.facebook.swift.codec.*;
import com.facebook.swift.service.*;
import com.facebook.swift.service.metadata.*;
import com.facebook.swift.transport.client.*;
import com.facebook.swift.transport.util.FutureUtil;
import java.io.*;
import java.lang.reflect.Method;
import java.util.*;
import org.apache.thrift.ProtocolId;
import reactor.core.publisher.Mono;

@SwiftGenerated
public class MyServiceClientImpl extends AbstractThriftClient implements MyService {


    // Method Handlers
    private ThriftMethodHandler queryMethodHandler;
    private ThriftMethodHandler hasArgDocsMethodHandler;

    // Method Exceptions
    private static final Class[] queryExceptions = new Class[] {
        org.apache.thrift.TException.class};
    private static final Class[] hasArgDocsExceptions = new Class[] {
        org.apache.thrift.TException.class};

    public MyServiceClientImpl(
        RequestChannel channel,
        Map<Method, ThriftMethodHandler> methods,
        Map<String, String> headers,
        Map<String, String> persistentHeaders,
        List<? extends ThriftClientEventHandler> eventHandlers) {
      super(channel, headers, persistentHeaders, eventHandlers);

      Map<String, ThriftMethodHandler> methodHandlerMap = new HashMap<>();
      methods.forEach(
          (key, value) -> {
            methodHandlerMap.put(key.getName(), value);
          });

      // Set method handlers
      queryMethodHandler = methodHandlerMap.get("query");
      hasArgDocsMethodHandler = methodHandlerMap.get("hasArgDocs");
    }

    public MyServiceClientImpl(
        Map<String, String> headers,
        Map<String, String> persistentHeaders,
        Mono<? extends RpcClient> rpcClient,
        ThriftServiceMetadata serviceMetadata,
        ThriftCodecManager codecManager,
        ProtocolId protocolId,
        Map<Method, ThriftMethodHandler> methods) {
      super(headers, persistentHeaders, rpcClient, serviceMetadata, codecManager, protocolId);

      Map<String, ThriftMethodHandler> methodHandlerMap = new HashMap<>();
      methods.forEach(
          (key, value) -> {
            methodHandlerMap.put(key.getName(), value);
          });

      // Set method handlers
      queryMethodHandler = methodHandlerMap.get("query");
      hasArgDocsMethodHandler = methodHandlerMap.get("hasArgDocs");
    }

    @java.lang.Override
    public void close() {
        super.close();
    }


    @java.lang.Override
    public void query(
        test.fixtures.includes.MyStruct s,
        test.fixtures.includes.includes.Included i) throws org.apache.thrift.TException {
      queryWrapper(s, i, RpcOptions.EMPTY).getData();
    }

    @java.lang.Override
    public void query(
        test.fixtures.includes.MyStruct s,
        test.fixtures.includes.includes.Included i,
        RpcOptions rpcOptions) throws org.apache.thrift.TException {
      queryWrapper(s, i, rpcOptions).getData();
    }

    @java.lang.Override
    public ResponseWrapper<Void> queryWrapper(
        test.fixtures.includes.MyStruct s,
        test.fixtures.includes.includes.Included i,
        RpcOptions rpcOptions) throws org.apache.thrift.TException {
      try {
        return FutureUtil.get(executeWrapperWithOptions(queryMethodHandler, queryExceptions, rpcOptions, s, i));
      } catch (Throwable t) {
        if (t instanceof org.apache.thrift.TException) {
          throw (org.apache.thrift.TException) t;
        }
        throw new org.apache.thrift.TException(t);
      }
    }

    @java.lang.Override
    public void hasArgDocs(
        test.fixtures.includes.MyStruct s,
        test.fixtures.includes.includes.Included i) throws org.apache.thrift.TException {
      hasArgDocsWrapper(s, i, RpcOptions.EMPTY).getData();
    }

    @java.lang.Override
    public void hasArgDocs(
        test.fixtures.includes.MyStruct s,
        test.fixtures.includes.includes.Included i,
        RpcOptions rpcOptions) throws org.apache.thrift.TException {
      hasArgDocsWrapper(s, i, rpcOptions).getData();
    }

    @java.lang.Override
    public ResponseWrapper<Void> hasArgDocsWrapper(
        test.fixtures.includes.MyStruct s,
        test.fixtures.includes.includes.Included i,
        RpcOptions rpcOptions) throws org.apache.thrift.TException {
      try {
        return FutureUtil.get(executeWrapperWithOptions(hasArgDocsMethodHandler, hasArgDocsExceptions, rpcOptions, s, i));
      } catch (Throwable t) {
        if (t instanceof org.apache.thrift.TException) {
          throw (org.apache.thrift.TException) t;
        }
        throw new org.apache.thrift.TException(t);
      }
    }
}
