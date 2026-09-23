# Build stage
FROM registry.access.redhat.com/ubi9/openjdk-25 AS builder

USER root
RUN microdnf install -y maven && microdnf clean all
RUN mkdir -p /app/trend-service/target && chown -R ${CONTAINER_DEFAULT_USER}:${CONTAINER_DEFAULT_USER} /app
USER ${CONTAINER_DEFAULT_USER}

WORKDIR /app

COPY pom.xml /app/trend-service/
COPY src /app/trend-service/src

RUN cd /app/trend-service && \
    mvn package -DskipTests -Dquarkus.package.type=fast-jar

# Runtime stage
FROM registry.access.redhat.com/hi/openjdk:latest-runtime

WORKDIR /app

COPY --from=builder /app/trend-service/target/quarkus-app /app/trend-service/target/quarkus-app/

ENV JAVA_TOOL_OPTIONS="-Xms256m -Xmx512m"
ENV QUARKUS_HTTP_PORT=8081

EXPOSE 8081

CMD ["java", "-jar", "/app/trend-service/target/quarkus-app/quarkus-run.jar"]
