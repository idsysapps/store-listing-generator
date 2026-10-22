package com.trends.resolver;

import com.trends.config.ProductScope;
import io.quarkus.security.identity.SecurityIdentity;
import org.eclipse.microprofile.graphql.GraphQLException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.security.Principal;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ProductScopeResolverTest {

    @Mock
    ProductScope productScope;

    @Mock
    SecurityIdentity securityIdentity;

    @InjectMocks
    ProductScopeResolver resolver;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    private void asAuthenticatedUser() {
        Principal principal = () -> "operator@example.com";
        when(securityIdentity.isAnonymous()).thenReturn(false);
        when(securityIdentity.getPrincipal()).thenReturn(principal);
    }

    @Test
    void testGetEnabledProductTypes_ReturnsTheScope() throws GraphQLException {
        asAuthenticatedUser();
        when(productScope.enabledProductTypes()).thenReturn(List.of("dtf_apparel"));

        assertEquals(List.of("dtf_apparel"), resolver.getEnabledProductTypes());
    }

    @Test
    void testGetEnabledProductTypes_RequiresAuthentication() {
        when(securityIdentity.isAnonymous()).thenReturn(true);

        assertThrows(GraphQLException.class, () -> resolver.getEnabledProductTypes());
    }

    @Test
    void testSetEnabledProductTypes_StoresTheScope() throws GraphQLException {
        asAuthenticatedUser();
        when(productScope.enabledProductTypes()).thenReturn(List.of("dtf_apparel"));
        doNothing().when(productScope).setEnabledProductTypes(List.of("dtf_apparel"), "operator@example.com");

        assertEquals(List.of("dtf_apparel"), resolver.setEnabledProductTypes(List.of("dtf_apparel")));

        verify(productScope).setEnabledProductTypes(List.of("dtf_apparel"), "operator@example.com");
    }

    @Test
    void testSetEnabledProductTypes_RejectsAnEmptyList() {
        asAuthenticatedUser();

        GraphQLException error = assertThrows(GraphQLException.class,
                () -> resolver.setEnabledProductTypes(List.of()));

        assertTrue(error.getMessage().contains("At least one product type"));
    }

    @Test
    void testSetEnabledProductTypes_RejectsAnUnknownType() {
        asAuthenticatedUser();

        GraphQLException error = assertThrows(GraphQLException.class,
                () -> resolver.setEnabledProductTypes(List.of("screen_print")));

        assertTrue(error.getMessage().contains("screen_print"));
    }

    @Test
    void testSetEnabledProductTypes_RequiresAuthentication() {
        when(securityIdentity.isAnonymous()).thenReturn(true);

        assertThrows(GraphQLException.class,
                () -> resolver.setEnabledProductTypes(List.of("dtf_apparel")));
    }
}
