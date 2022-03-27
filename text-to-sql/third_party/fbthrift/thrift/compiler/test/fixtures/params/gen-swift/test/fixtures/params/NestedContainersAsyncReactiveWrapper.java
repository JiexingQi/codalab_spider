/**
 * Autogenerated by Thrift
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */

package test.fixtures.params;

import java.util.*;

public class NestedContainersAsyncReactiveWrapper 
    implements NestedContainers.Reactive {
    private final NestedContainers.Async _delegate;

    public NestedContainersAsyncReactiveWrapper(NestedContainers.Async _delegate) {
        
        this._delegate = _delegate;
    }

    @java.lang.Override
    public void dispose() {
        _delegate.close();
    }

    @java.lang.Override
    public reactor.core.publisher.Mono<Void> mapList(final Map<Integer, List<Integer>> foo) {
        return com.facebook.swift.transport.util.FutureUtil.toMono(_delegate.mapList(foo));
    }

    @java.lang.Override
    public reactor.core.publisher.Mono<Void> mapSet(final Map<Integer, Set<Integer>> foo) {
        return com.facebook.swift.transport.util.FutureUtil.toMono(_delegate.mapSet(foo));
    }

    @java.lang.Override
    public reactor.core.publisher.Mono<Void> listMap(final List<Map<Integer, Integer>> foo) {
        return com.facebook.swift.transport.util.FutureUtil.toMono(_delegate.listMap(foo));
    }

    @java.lang.Override
    public reactor.core.publisher.Mono<Void> listSet(final List<Set<Integer>> foo) {
        return com.facebook.swift.transport.util.FutureUtil.toMono(_delegate.listSet(foo));
    }

    @java.lang.Override
    public reactor.core.publisher.Mono<Void> turtles(final List<List<Map<Integer, Map<Integer, Set<Integer>>>>> foo) {
        return com.facebook.swift.transport.util.FutureUtil.toMono(_delegate.turtles(foo));
    }

}
