package com.trends.config;

import com.trends.domain.AppSetting;
import com.trends.repository.AppSettingRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.lang.reflect.Field;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

class ProductScopeTest {

    @Mock
    AppSettingRepository appSettingRepository;

    @InjectMocks
    ProductScope scope;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field envField = ProductScope.class.getDeclaredField("envValue");
        envField.setAccessible(true);
        envField.set(scope, "");
    }

    private void withEnvironment(String value) throws Exception {
        Field envField = ProductScope.class.getDeclaredField("envValue");
        envField.setAccessible(true);
        envField.set(scope, value);
    }

    private void withStoredScope(String value) {
        if (value == null) {
            when(appSettingRepository.findByKey(ProductScope.SETTING_KEY)).thenReturn(null);
            return;
        }
        AppSetting setting = new AppSetting();
        setting.key = ProductScope.SETTING_KEY;
        setting.value = value;
        when(appSettingRepository.findByKey(ProductScope.SETTING_KEY)).thenReturn(setting);
    }

    @Test
    void testEnabledProductTypes_WithoutAScopeIsEveryKnownType() {
        withStoredScope(null);

        assertEquals(List.of("dtf_apparel", "sublimation", "sticker_vinyl"),
                scope.enabledProductTypes());
    }

    @Test
    void testEnabledProductTypes_TheEnvironmentNamesTheTypes() throws Exception {
        withStoredScope(null);
        withEnvironment("dtf_apparel");

        assertEquals(List.of("dtf_apparel"), scope.enabledProductTypes());
    }

    @Test
    void testEnabledProductTypes_TheStoredScopeWinsOverTheEnvironment() throws Exception {
        withStoredScope("sublimation");
        withEnvironment("dtf_apparel");

        assertEquals(List.of("sublimation"), scope.enabledProductTypes());
    }

    @Test
    void testEnabledProductTypes_AStoredScopeThatNamesNothingFallsBackToTheEnvironment() throws Exception {
        withStoredScope("screen_print");
        withEnvironment("dtf_apparel");

        assertEquals(List.of("dtf_apparel"), scope.enabledProductTypes());
    }

    @Test
    void testEnabledProductTypes_AnEmptyStoredScopeFallsBackToTheEnvironment() throws Exception {
        withStoredScope("");
        withEnvironment("dtf_apparel");

        assertEquals(List.of("dtf_apparel"), scope.enabledProductTypes());
    }

    @Test
    void testEnabledProductTypes_TheScopeIsReadByItsKey() {
        withStoredScope("dtf_apparel");

        scope.enabledProductTypes();

        org.mockito.Mockito.verify(appSettingRepository).findByKey(ProductScope.SETTING_KEY);
    }

    @Test
    void testEnabledProductTypes_AScopeThatCannotBeReadFallsBackToTheEnvironment() throws Exception {
        withEnvironment("dtf_apparel");
        when(appSettingRepository.findByKey(ProductScope.SETTING_KEY))
                .thenThrow(new RuntimeException("the table is not there yet"));

        assertEquals(List.of("dtf_apparel"), scope.enabledProductTypes());
    }

    @Test
    void testParse_DropsAnUnknownType() {
        assertEquals(List.of("dtf_apparel"), scope.parse("dtf_apparel, screen_print"));
    }

    @Test
    void testParse_DropsARepeatedType() {
        assertEquals(List.of("dtf_apparel"), scope.parse("dtf_apparel,dtf_apparel"));
    }

    @Test
    void testParse_ReadsACommaSeparatedList() {
        assertEquals(List.of("dtf_apparel", "sublimation"), scope.parse("dtf_apparel, sublimation"));
    }
}
