package com.trends.domain;

import org.hibernate.SessionFactory;
import org.hibernate.boot.MetadataSources;
import org.hibernate.boot.registry.StandardServiceRegistry;
import org.hibernate.boot.registry.StandardServiceRegistryBuilder;
import org.hibernate.engine.spi.SessionFactoryImplementor;
import org.hibernate.metamodel.spi.MappingMetamodelImplementor;
import org.hibernate.persister.entity.EntityPersister;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Guards the nullability of the columns cost-book may leave out of a material event.
 *
 * <p>Cost-book publishes a field only when the material carries it, so an event for a
 * material without a product type has no {@code substrate_type} and no
 * {@code product_type}. Both columns therefore have to be nullable - the behaviour is
 * covered by {@code FabricationMaterialJsonPersistenceIT} against a real PostgreSQL, this
 * test keeps the mapping honest without needing a database.
 */
class FabricationMaterialNullabilityTest {

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
    void shouldMapSubstrateTypeAsNullable() {
        assertTrue(isNullable("substrateType"),
                "fabrication_materials.substrate_type is optional in the event contract");
    }

    @Test
    void shouldMapProductTypeAsNullable() {
        assertTrue(isNullable("productType"),
                "fabrication_materials.product_type is optional in the event contract");
    }

    private static boolean isNullable(String attribute) {
        EntityPersister persister = sessionFactory
                .unwrap(SessionFactoryImplementor.class)
                .getMappingMetamodel()
                .getEntityDescriptor(FabricationMaterial.class);
        String[] propertyNames = persister.getPropertyNames();
        boolean[] nullability = persister.getPropertyNullability();
        int index = Arrays.asList(propertyNames).indexOf(attribute);
        assertTrue(index >= 0, "unknown property " + attribute);
        return nullability[index];
    }
}
