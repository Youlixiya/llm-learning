// Enhanced Markdown Loader with Base URL Support
(function() {
  'use strict';

  // Get base path for GitHub Pages
  function getBasePath() {
    const path = window.location.pathname;
    // If path starts with /llm-learning/ or similar, extract it
    const match = path.match(/^(\/[^\/]+\/)/);
    if (match && match[1] !== '/') {
      return match[1];
    }
    return '';
  }

  // Load markdown file
  function loadMarkdown(mdPath, targetId) {
    const target = document.getElementById(targetId);
    if (!target) {
      console.error('Target element not found:', targetId);
      return;
    }

    // Try multiple path variations
    const basePath = getBasePath();
    const paths = [
      mdPath, // Original path
      basePath + mdPath.replace(/^\.\.\//, ''), // With base path, remove ../
      mdPath.replace(/^\.\.\//, ''), // Without ../
    ];

    let currentPathIndex = 0;

    function tryLoadPath() {
      if (currentPathIndex >= paths.length) {
        target.classList.remove('loading');
        target.classList.add('error');
        target.innerHTML = `
          <p>无法加载文档内容。请检查以下路径：</p>
          <ul>
            ${paths.map(p => `<li><code>${p}</code></li>`).join('')}
          </ul>
          <p>如果问题持续存在，请<a href="https://github.com/youlixiya/llm-learning/issues" target="_blank">提交 Issue</a>。</p>
        `;
        return;
      }

      const currentPath = paths[currentPathIndex];
      console.log('Trying to load:', currentPath);

      fetch(currentPath)
        .then((res) => {
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }
          return res.text();
        })
        .then((text) => {
          try {
            if (typeof marked !== 'undefined') {
              const html = marked.parse(text);
              target.classList.remove('loading');
              target.classList.remove('error');
              target.innerHTML = html;
              console.log('Successfully loaded:', currentPath);
            } else {
              throw new Error('Marked.js library not loaded');
            }
          } catch (e) {
            console.error('Error parsing markdown:', e);
            target.classList.remove('loading');
            target.classList.add('error');
            target.textContent = '渲染 Markdown 时出错：' + e.message;
          }
        })
        .catch((err) => {
          console.warn('Failed to load:', currentPath, err);
          currentPathIndex++;
          tryLoadPath();
        });
    }

    tryLoadPath();
  }

  // Auto-load markdown for elements with data-md-path attribute
  document.addEventListener('DOMContentLoaded', function() {
    const elements = document.querySelectorAll('[data-md-path]');
    elements.forEach(function(el) {
      const mdPath = el.getAttribute('data-md-path');
      const targetId = el.getAttribute('data-md-target') || 'chapter-md';
      loadMarkdown(mdPath, targetId);
    });
  });

  // Export function for manual use
  window.loadMarkdown = loadMarkdown;
})();
