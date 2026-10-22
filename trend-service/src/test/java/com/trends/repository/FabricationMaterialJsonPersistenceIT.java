package com.trends.repository;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.trends.domain.FabricationMaterial;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.hibernate.boot.MetadataSources;
import org.hibernate.boot.registry.StandardServiceRegistry;
import org.hibernate.boot.registry.StandardServiceRegistryBuilder;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.testcontainers.postgresql.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.OffsetDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Persistence of {@link FabricationMaterial} against a real PostgreSQL.
 *
 * <p>{@code attributes} is a {@code jsonb} column in production. H2 cannot create
 * {@code jsonb} columns, so the failure this covers - a varchar parameter being sent
 * to a jsonb column - only shows up against PostgreSQL. Hence the container.
 */
@Testcontainers
class FabricationMaterialJsonPersistenceIT {

    private static final String ATTRIBUTES = "{\"color_code\":\"#FF0000\",\"weight_g\":180}";

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Container
    private static final PostgreSQLContainer POSTGRES = new PostgreSQLContainer(
            DockerImageName.parse("docker.io/library/postgres:16-alpine").asCompatibleSubstituteFor("postgres"))
            .withDatabaseName("store_listing")
            .withUsername("store_listing")
            .withPassword("store_listing");

    private static StandardServiceRegistry registry;
    private static SessionFactory sessionFactory;

    @BeforeAll
    static void setUp() throws IOException, SQLException {
        createSchema();

        registry = new StandardServiceRegistryBuilder()
                .applySetting("hibernate.connection.driver_class", "org.postgresql.Driver")
                .applySetting("hibernate.connection.url", POSTGRES.getJdbcUrl())
                .applySetting("hibernate.connection.username", POSTGRES.getUsername())
                .applySetting("hibernate.connection.password", POSTGRES.getPassword())
                .applySetting("hibernate.dialect", "org.hibernate.dialect.PostgreSQLDialect")
                .applySetting("hibernate.hbm2ddl.auto", "none")
                .build();
        sessionFactory = new MetadataSources(registry)
                .addAnnotatedClass(FabricationMaterial.class)
                .buildMetadata()
                .buildSessionFactory();
    }

    @AfterAll
    static void tearDown() {
        if (sessionFactory != null) {
            sessionFactory.close();
        }
        if (registry != null) {
            StandardServiceRegistryBuilder.destroy(registry);
        }
    }

    @BeforeEach
    void clearMaterials() throws SQLException {
        try (Connection connection = openConnection();
             Statement statement = connection.createStatement()) {
            statement.execute("DELETE FROM fabrication_materials");
        }
    }

    @Test
    void shouldPersistMaterialWithJsonAttributes() throws SQLException {
        Integer id = persistMaterial(ATTRIBUTES);

        FabricationMaterial saved = findById(id);
        assertNotNull(saved, "material should have been persisted");
        assertSameJson(ATTRIBUTES, saved.attributes, "attributes should round-trip unchanged");

        StoredAttributes stored = readStoredAttributes(id);
        assertEquals("jsonb", stored.columnType(), "attributes should be stored as jsonb");
        assertEquals("#FF0000", stored.colorCode(), "attributes should be stored as a JSON document");
    }

    @Test
    void shouldUpdateJsonAttributesOfExistingMaterial() throws SQLException {
        Integer id = persistMaterial(ATTRIBUTES);

        try (Session session = sessionFactory.openSession()) {
            Transaction transaction = session.beginTransaction();
            FabricationMaterial material = session.get(FabricationMaterial.class, id);
            material.attributes = "{\"color_code\":\"#00FF00\"}";
            material.updatedAt = OffsetDateTime.now();
            transaction.commit();
        }

        FabricationMaterial saved = findById(id);
        assertNotNull(saved, "material should still exist");
        assertSameJson("{\"color_code\":\"#00FF00\"}", saved.attributes, "attributes should be replaced");

        StoredAttributes stored = readStoredAttributes(id);
        assertEquals("jsonb", stored.columnType(), "attributes should be stored as jsonb");
        assertEquals("#00FF00", stored.colorCode(), "attributes should be stored as a JSON document");
    }

    @Test
    void shouldPersistMaterialWithoutAttributes() throws SQLException {
        Integer id = persistMaterial(null);

        FabricationMaterial saved = findById(id);
        assertNotNull(saved, "material should have been persisted");
        assertNull(saved.attributes, "attributes should stay null when the event carries none");
    }

    @Test
    void shouldPersistMaterialWithoutSubstrateTypeAndProductType() throws SQLException {
        Integer id = persistMaterialWithoutOptionalFields();

        FabricationMaterial saved = findById(id);
        assertNotNull(saved, "material should have been persisted");
        assertNull(saved.substrateType, "substrate_type should stay null when the event carries none");
        assertNull(saved.productType, "product_type should stay null when the event carries none");
        assertEquals("Core Adult", saved.name, "the fields the event does carry should be stored");
    }

    @Test
    void shouldPersistMaterialSizesAsTextArray() throws SQLException {
        Integer id = persistMaterial(ATTRIBUTES);

        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(
                     "SELECT sizes FROM fabrication_materials WHERE id = ?")) {
            statement.setInt(1, id);
            try (ResultSet result = statement.executeQuery()) {
                assertNotNull(result.next(), "material row should exist");
                Object[] sizes = (Object[]) result.getArray("sizes").getArray();
                assertEquals(List.of("S", "M", "L"), List.of(sizes));
            }
        }
    }

    @Test
    void shouldPersistProductTypesAsTextArray() throws SQLException {
        // Cost-book publishes the product types a material can serve as a JSON
        // array, so the column holds the array (issue #186).
        Integer id = persistMaterial(ATTRIBUTES);

        FabricationMaterial saved = findById(id);
        assertNotNull(saved, "material should have been persisted");
        assertEquals(List.of("substrate", "dtf_apparel"), List.of(saved.productType));

        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(
                     "SELECT product_type FROM fabrication_materials WHERE id = ?")) {
            statement.setInt(1, id);
            try (ResultSet result = statement.executeQuery()) {
                assertNotNull(result.next(), "material row should exist");
                Object[] productTypes = (Object[]) result.getArray("product_type").getArray();
                assertEquals(List.of("substrate", "dtf_apparel"), List.of(productTypes));
            }
        }
    }

    @Test
    void shouldFindMaterialsByAnyOfTheirProductTypes() throws SQLException {
        persistMaterial(ATTRIBUTES);

        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(
                     "SELECT external_id FROM fabrication_materials "
                             + "WHERE active = true AND ? = ANY(product_type)")) {
            statement.setString(1, "dtf_apparel");
            try (ResultSet result = statement.executeQuery()) {
                assertNotNull(result.next(), "the material serves dtf_apparel");
                assertEquals("bc3001-white", result.getString("external_id"));
            }
        }

        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(
                     "SELECT external_id FROM fabrication_materials "
                             + "WHERE active = true AND ? = ANY(product_type)")) {
            statement.setString(1, "sublimation");
            try (ResultSet result = statement.executeQuery()) {
                assertFalse(result.next(), "the material does not serve sublimation");
            }
        }
    }

    /**
     * jsonb normalises the document it stores (key order, whitespace), so the comparison
     * has to be made on the parsed documents.
     */
    private static void assertSameJson(String expected, String actual, String message) {
        assertNotNull(actual, message);
        try {
            assertEquals(OBJECT_MAPPER.readTree(expected), OBJECT_MAPPER.readTree(actual), message);
        } catch (JsonProcessingException e) {
            throw new AssertionError("not valid JSON: " + actual, e);
        }
    }

    private static Integer persistMaterial(String attributes) {
        FabricationMaterial material = new FabricationMaterial();
        material.externalId = "bc3001-white";
        material.name = "Bella Canvas 3001";
        material.brand = "Bella Canvas";
        material.substrateType = "t-shirt";
        material.color = "White";
        material.sizes = new String[]{"S", "M", "L"};
        material.materialComposition = "100% ring-spun cotton";
        material.productType = new String[] { "substrate", "dtf_apparel" };
        material.sku = "BC3001-WHT";
        material.unitCost = new BigDecimal("4.50");
        material.active = true;
        material.attributes = attributes;
        material.createdAt = OffsetDateTime.now();
        material.updatedAt = OffsetDateTime.now();

        try (Session session = sessionFactory.openSession()) {
            Transaction transaction = session.beginTransaction();
            session.persist(material);
            transaction.commit();
        }
        return material.id;
    }

    private static Integer persistMaterialWithoutOptionalFields() {
        FabricationMaterial material = new FabricationMaterial();
        material.externalId = "c9924c1b-78a6-4a38-9bb0-32c44cc4eb1a-true-celadon";
        material.name = "Core Adult";
        material.brand = "Port & Company";
        material.color = "True Celadon";
        material.sku = "PC54-TRUE-CELADON-L";
        material.unitCost = new BigDecimal("2.39");
        material.active = true;
        material.createdAt = OffsetDateTime.now();
        material.updatedAt = OffsetDateTime.now();

        try (Session session = sessionFactory.openSession()) {
            Transaction transaction = session.beginTransaction();
            session.persist(material);
            transaction.commit();
        }
        return material.id;
    }

    private static FabricationMaterial findById(Integer id) {
        try (Session session = sessionFactory.openSession()) {
            return session.get(FabricationMaterial.class, id);
        }
    }

    private static void createSchema() throws IOException, SQLException {
        try (Connection connection = openConnection();
             Statement statement = connection.createStatement()) {
            for (String ddl : readSchema().split(";")) {
                if (!ddl.trim().isEmpty()) {
                    statement.execute(ddl);
                }
            }
        }
    }

    private static Connection openConnection() throws SQLException {
        return DriverManager.getConnection(
                POSTGRES.getJdbcUrl(), POSTGRES.getUsername(), POSTGRES.getPassword());
    }

    private static String readSchema() throws IOException {
        try (InputStream schema = FabricationMaterialJsonPersistenceIT.class
                .getResourceAsStream("/db/fabrication_materials.sql")) {
            if (schema == null) {
                throw new IllegalStateException("missing test schema /db/fabrication_materials.sql");
            }
            return new String(schema.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static StoredAttributes readStoredAttributes(Integer id) throws SQLException {
        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(
                     "SELECT pg_typeof(attributes)::text AS column_type, attributes->>'color_code' AS color_code "
                             + "FROM fabrication_materials WHERE id = ?")) {
            statement.setInt(1, id);
            try (ResultSet result = statement.executeQuery()) {
                assertNotNull(result.next(), "material row should exist");
                return new StoredAttributes(result.getString("column_type"), result.getString("color_code"));
            }
        }
    }

    private record StoredAttributes(String columnType, String colorCode) {
    }
}
