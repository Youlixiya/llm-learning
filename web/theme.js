// Theme Toggle Functionality
(function() {
  'use strict';

  const THEME_KEY = 'llm-tutorial-theme';
  const THEME_ATTRIBUTE = 'data-theme';

  // Get current theme from localStorage or default to 'dark'
  function getTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'light' || saved === 'dark') {
      return saved;
    }
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      return 'light';
    }
    return 'dark';
  }

  // Set theme
  function setTheme(theme) {
    if (theme !== 'light' && theme !== 'dark') {
      return;
    }
    document.documentElement.setAttribute(THEME_ATTRIBUTE, theme);
    localStorage.setItem(THEME_KEY, theme);
    updateThemeButton(theme);
  }

  // Update theme button icon
  function updateThemeButton(theme) {
    const buttons = document.querySelectorAll('.theme-toggle');
    buttons.forEach(button => {
      button.textContent = theme === 'light' ? '🌙' : '☀️';
      button.setAttribute('aria-label', theme === 'light' ? '切换到夜间模式' : '切换到白天模式');
    });
  }

  // Toggle theme
  function toggleTheme() {
    const current = getTheme();
    const newTheme = current === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
  }

  // Initialize theme on page load
  function initTheme() {
    const theme = getTheme();
    setTheme(theme);
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTheme);
  } else {
    initTheme();
  }

  // Expose toggle function globally
  window.toggleTheme = toggleTheme;

  // Listen for system theme changes
  if (window.matchMedia) {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
    mediaQuery.addEventListener('change', (e) => {
      // Only auto-switch if user hasn't manually set a preference
      if (!localStorage.getItem(THEME_KEY)) {
        setTheme(e.matches ? 'light' : 'dark');
      }
    });
  }
})();
