# Build stage
FROM registry.access.redhat.com/hi/openjdk:25-builder AS builder

RUN microdnf install -y maven && microdnf clean all

WORKDIR /app

COPY pom.xml /app/trend-service/
COPY src /app/trend-service/src

RUN cd /app/trend-service && \
    mvn package -DskipTests -Dquarkus.package.type=fast-jar

# Runtime stage
FROM registry.access.redhat.com/hi/openjdk:25

WORKDIR /app

COPY --from=builder /app/trend-service/target/quarkus-app /app/trend-service/target/quarkus-app/

ENV JAVA_OPTS="-Xms256m -Xmx512m"
ENV QUARKUS_HTTP_PORT=8081

EXPOSE 8081

CMD ["/app/trend-service/target/quarkus-app/bin/run"]
