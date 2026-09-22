package com.trends.event;

import com.trends.dto.TrendUpdate;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.subscription.Cancellable;
import io.smallrye.mutiny.subscription.MultiSubscriber;
import io.smallrye.mutiny.subscription.SerializedSubscriber;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import java.util.concurrent.CopyOnWriteArrayList;

@ApplicationScoped
public class InMemoryTrendingEventPublisher implements TrendingEventPublisher {

    private static final Logger LOG = Logger.getLogger(InMemoryTrendingEventPublisher.class);

    private final CopyOnWriteArrayList<SerializedSubscriber<TrendUpdate>> subscribers = new CopyOnWriteArrayList<>();

    @Override
    public void publish(TrendUpdate update) {
        LOG.infof("Publishing trend update: query=%s, delta=%d, score=%d",
                update.getQuery(), update.getDelta(), update.getScore());

        for (SerializedSubscriber<TrendUpdate> subscriber : subscribers) {
            try {
                subscriber.onItem(update);
            } catch (Exception e) {
                LOG.errorf(e, "Error publishing to subscriber");
            }
        }
    }

    @Override
    public Multi<TrendUpdate> subscribe() {
        return Multi.createFrom(). emitter(emitter -> {
            SerializedSubscriber<TrendUpdate> subscriber = new SerializedSubscriber<>(new MultiSubscriber<>() {
                @Override
                public void onSubscribe(Cancellable cancellable) {
                    subscribers.add(subscriber);
                }

                @Override
                public void onItem(TrendUpdate item) {
                    emitter.emit(item);
                }

                @Override
                public void onFailure(Throwable error) {
                    emitter.fail(error);
                    subscribers.remove(subscriber);
                }

                @Override
                public void onCompletion() {
                    subscribers.remove(subscriber);
                }
            });
        });
    }
}
