package com.trends.repository;

import com.trends.domain.SourceHealth;
import io.quarkus.hibernate.orm.panache.PanacheRepositoryBase;
import jakarta.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class SourceHealthRepository implements PanacheRepositoryBase<SourceHealth, String> {
}
