package com.trends.event;

import com.trends.dto.TrendUpdate;
import io.smallrye.mutiny.Multi;

public interface TrendingEventPublisher {

    void publish(TrendUpdate update);

    Multi<TrendUpdate> subscribe();
}
