package com.trends.event;

import com.trends.dto.TrendUpdate;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.subscription.MultiSubscriber;
import io.smallrye.mutiny.subscription.SerializedSubscriber;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Flow.Subscription;

@ApplicationScoped
public class InMemoryTrendingEventPublisher implements TrendingEventPublisher {

    private static final Logger LOG = Logger.getLogger(InMemoryTrendingEventPublisher.class);

    private final ConcurrentHashMap<String, SerializedSubscriber<TrendUpdate>> subscribers = new ConcurrentHashMap<>();

    @Override
    public void publish(TrendUpdate update) {
        LOG.infof("Publishing trend update: query=%s, delta=%d, score=%d",
                update.getQuery(), update.getDelta(), update.getScore());

        subscribers.values().forEach(subscriber -> {
            try {
                subscriber.onItem(update);
            } catch (Exception e) {
                LOG.errorf(e, "Error publishing to subscriber");
            }
        });
    }

    @Override
    public Multi<TrendUpdate> subscribe() {
        return Multi.createFrom(). emitter(emitter -> {
            String subscriberId = Integer.toHexString(System.identityHashCode(emitter));
            SerializedSubscriber<TrendUpdate> subscriber = new SerializedSubscriber<>(createSubscriber(emitter, subscriberId));
            subscribers.put(subscriberId, subscriber);
        });
    }

    private MultiSubscriber<TrendUpdate> createSubscriber(io.smallrye.mutiny.subscription.MultiEmitter<? super TrendUpdate> emitter, String subscriberId) {
        return new MultiSubscriber<>() {
            @Override
            public void onSubscribe(Subscription subscription) {
            }

            @Override
            public void onItem(TrendUpdate item) {
                emitter.emit(item);
            }

            @Override
            public void onFailure(Throwable error) {
                emitter.fail(error);
                subscribers.remove(subscriberId);
            }

            @Override
            public void onCompletion() {
                subscribers.remove(subscriberId);
            }
        };
    }
}
