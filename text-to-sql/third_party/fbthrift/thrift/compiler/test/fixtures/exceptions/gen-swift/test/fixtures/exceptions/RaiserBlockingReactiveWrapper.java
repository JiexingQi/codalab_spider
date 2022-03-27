/**
 * Autogenerated by Thrift
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */

package test.fixtures.exceptions;

import java.util.*;

public class RaiserBlockingReactiveWrapper 
    implements Raiser.Reactive {
    private final Raiser _delegate;
    private final reactor.core.scheduler.Scheduler _scheduler;

    public RaiserBlockingReactiveWrapper(Raiser _delegate, reactor.core.scheduler.Scheduler _scheduler) {
        
        this._delegate = _delegate;
        this._scheduler = _scheduler;
    }

    @java.lang.Override
    public void dispose() {
        _delegate.close();
    }

    @java.lang.Override
    public reactor.core.publisher.Mono<Void> doBland() {
        return reactor.core.publisher.Mono.<Void>fromRunnable(() -> {
                try {
                    _delegate.doBland();
                } catch (Throwable _e) {
                    throw reactor.core.Exceptions.propagate(_e);
                }
            }).subscribeOn(_scheduler);
    }

    @java.lang.Override
    public reactor.core.publisher.Mono<Void> doRaise() {
        return reactor.core.publisher.Mono.<Void>fromRunnable(() -> {
                try {
                    _delegate.doRaise();
                } catch (Throwable _e) {
                    throw reactor.core.Exceptions.propagate(_e);
                }
            }).subscribeOn(_scheduler);
    }

    @java.lang.Override
    public reactor.core.publisher.Mono<String> get200() {
        return reactor.core.publisher.Mono.fromSupplier(() -> {
                try {
                    return _delegate.get200();
                } catch (Throwable _e) {
                    throw reactor.core.Exceptions.propagate(_e);
                }
            }).subscribeOn(_scheduler);
    }

    @java.lang.Override
    public reactor.core.publisher.Mono<String> get500() {
        return reactor.core.publisher.Mono.fromSupplier(() -> {
                try {
                    return _delegate.get500();
                } catch (Throwable _e) {
                    throw reactor.core.Exceptions.propagate(_e);
                }
            }).subscribeOn(_scheduler);
    }

}
