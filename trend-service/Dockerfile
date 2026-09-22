FROM maven:3.9-eclipse-temurin-17 AS builder

WORKDIR /app

COPY trend-service/pom.xml /app/trend-service/
COPY trend-service/src /app/trend-service/src

RUN cd /app/trend-service && \
    mvn package -DskipTests -Dquarkus.package.type=fast-jar

FROM registry.access.redhat.com/ubi9/ubi-minimal:9.4

RUN microdnf install -y java-17-runtime && \
    microdnf clean all

WORKDIR /app

COPY --from=builder /app/trend-service/target/quarkus-app /app/trend-service/target/quarkus-app/

ENV JAVA_OPTS="-Xms256m -Xmx512m"
ENV QUARKUS_HTTP_PORT=8081

EXPOSE 8081

CMD ["/app/trend-service/target/quarkus-app/bin/run"]
