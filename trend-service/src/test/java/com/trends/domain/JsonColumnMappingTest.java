package com.trends.domain;

import org.hibernate.SessionFactory;
import org.hibernate.boot.MetadataSources;
import org.hibernate.boot.registry.StandardServiceRegistry;
import org.hibernate.boot.registry.StandardServiceRegistryBuilder;
import org.hibernate.engine.spi.SessionFactoryImplementor;
import org.hibernate.metamodel.spi.MappingMetamodelImplementor;
import org.hibernate.persister.entity.EntityPersister;
import org.hibernate.type.BasicType;
import org.hibernate.type.descriptor.jdbc.JsonJdbcType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Guards the mapping of the {@code jsonb} columns.
 *
 * <p>A {@code String} attribute that is stored in a {@code jsonb} column has to be mapped
 * as JSON. Without it Hibernate binds the value as a varchar and PostgreSQL rejects the
 * statement with {@code column "attributes" is of type jsonb but expression is of type
 * character varying}. The behaviour is covered by
 * {@code FabricationMaterialJsonPersistenceIT} against a real PostgreSQL; this test keeps
 * the mapping honest without needing a database.
 */
class JsonColumnMappingTest {

    private static StandardServiceRegistry registry;
    private static SessionFactory sessionFactory;

    @BeforeAll
    static void buildMapping() {
        registry = new StandardServiceRegistryBuilder()
                .applySetting("hibernate.dialect", "org.hibernate.dialect.PostgreSQLDialect")
                .applySetting("hibernate.boot.allow_jdbc_metadata_access", "false")
                .build();
        sessionFactory = new MetadataSources(registry)
                .addAnnotatedClass(FabricationMaterial.class)
                .addAnnotatedClass(DesignBrief.class)
                .addAnnotatedClass(DesignBriefSource.class)
                .addAnnotatedClass(ActiveSeed.class)
                .addAnnotatedClass(SeedProductTag.class)
                .addAnnotatedClass(SourceHealth.class)
                .addAnnotatedClass(TrendQuery.class)
                .addAnnotatedClass(TrendScore.class)
                .buildMetadata()
                .buildSessionFactory();
    }

    @AfterAll
    static void closeMapping() {
        if (sessionFactory != null) {
            sessionFactory.close();
        }
        if (registry != null) {
            StandardServiceRegistryBuilder.destroy(registry);
        }
    }

    @Test
    void shouldBindMaterialAttributesAsJson() {
        assertTrue(isJson(FabricationMaterial.class, "attributes"),
                "fabrication_materials.attributes is a jsonb column and has to be bound as JSON");
    }

    @Test
    void shouldBindRecommendedMaterialsAsJson() {
        assertTrue(isJson(DesignBrief.class, "recommendedMaterials"),
                "design_briefs.recommended_materials is a jsonb column and has to be bound as JSON");
    }

    private static boolean isJson(Class<?> entity, String attribute) {
        MappingMetamodelImplementor metamodel = sessionFactory
                .unwrap(SessionFactoryImplementor.class)
                .getMappingMetamodel();
        EntityPersister persister = metamodel.getEntityDescriptor(entity);
        BasicType<?> type = (BasicType<?>) persister.getPropertyType(attribute);
        return type.getJdbcType() instanceof JsonJdbcType;
    }
}
