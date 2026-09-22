package com.trends.subscription;

import com.trends.dto.TrendUpdate;
import com.trends.event.TrendingEventPublisher;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

@ApplicationScoped
public class TrendSubscription {

    private static final Logger LOG = Logger.getLogger(TrendSubscription.class);

    @Inject
    TrendingEventPublisher eventPublisher;

    public Multi<TrendUpdate> trendUpdated() {
        LOG.info("New subscription to trend updates");
        return eventPublisher.subscribe();
    }
}
