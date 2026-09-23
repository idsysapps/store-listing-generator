package com.trends.resolver;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.security.TestSecurity;
import io.restassured.http.ContentType;
import org.junit.jupiter.api.Test;

import static io.restassured.RestAssured.given;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;

@QuarkusTest
class TrendGraphQLApiIT {

    @Test
    @TestSecurity(user = "admin", roles = "admin")
    void shouldExposeTrendQueriesQuery() {
        given()
                .contentType(ContentType.JSON)
                .body("""
                        {"query":"query { trendQueries(limit: 10, offset: 0) { edges { cursor } pageInfo { totalCount hasNextPage hasPreviousPage } } }"}
                        """)
                .when()
                .post("/graphql")
                .then()
                .statusCode(200)
                .body("data.trendQueries.pageInfo.totalCount", is(0))
                .body("data.trendQueries.pageInfo.hasNextPage", is(false))
                .body("data.trendQueries.pageInfo.hasPreviousPage", is(false));
    }

    @Test
    @TestSecurity(user = "admin", roles = "admin")
    void shouldExposeTopTrendsQuery() {
        given()
                .contentType(ContentType.JSON)
                .body("""
                        {"query":"query { topTrends(limit: 5) { query latestScore } }"}
                        """)
                .when()
                .post("/graphql")
                .then()
                .statusCode(200)
                .body("data.topTrends", notNullValue());
    }

    @Test
    void shouldRejectUnauthenticatedQuery() {
        given()
                .contentType(ContentType.JSON)
                .body("""
                        {"query":"query { topTrends(limit: 5) { query } }"}
                        """)
                .when()
                .post("/graphql")
                .then()
                .statusCode(200)
                .body("data.topTrends", nullValue())
                .body("errors[0].message", notNullValue());
    }
}