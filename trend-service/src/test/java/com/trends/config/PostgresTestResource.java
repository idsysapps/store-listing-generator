package com.trends.config;

import io.quarkus.test.common.QuarkusTestResourceLifecycleManager;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;

import java.util.Map;

/**
 * The PostgreSQL the integration tests run against.
 *
 * <p>H2 cannot create the {@code text[]} and {@code jsonb} columns the
 * entities declare, so the schema the tests worked with was not the schema
 * the code expects, and the integration test could not boot at all
 * (issue #181). The container gives the tests the real database.
 */
public class PostgresTestResource implements QuarkusTestResourceLifecycleManager {

    private static final PostgreSQLContainer POSTGRES = new PostgreSQLContainer(
            DockerImageName.parse("docker.io/library/postgres:16-alpine").asCompatibleSubstituteFor("postgres"))
            .withDatabaseName("store_listing")
            .withUsername("store_listing")
            .withPassword("store_listing");

    @Override
    public Map<String, String> start() {
        POSTGRES.start();
        return Map.of(
                "quarkus.datasource.jdbc.url", POSTGRES.getJdbcUrl(),
                "quarkus.datasource.username", POSTGRES.getUsername(),
                "quarkus.datasource.password", POSTGRES.getPassword());
    }

    @Override
    public void stop() {
        POSTGRES.stop();
    }
}
