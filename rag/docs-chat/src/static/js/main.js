document.addEventListener('DOMContentLoaded', function () {
    // for mobile menu toggle
    const menuToggleButton = document.getElementById('mobileMenuToggle');
    const mobileNavMenu = document.getElementById('mobileNavMenu');

    if (menuToggleButton && mobileNavMenu) {
        menuToggleButton.addEventListener('click', function () {
            mobileNavMenu.classList.toggle('active');
            // Optional: Toggle ARIA attribute for accessibility
            const isExpanded = mobileNavMenu.classList.contains('active');
            menuToggleButton.setAttribute('aria-expanded', isExpanded);
        });
    }

    // Optional: Close mobile menu if user clicks outside of it
    document.addEventListener('click', function(event) {
        const isClickInsideNav = mobileNavMenu.contains(event.target);
        const isClickOnToggleButton = menuToggleButton.contains(event.target);

        if (mobileNavMenu.classList.contains('active') && !isClickInsideNav && !isClickOnToggleButton) {
            mobileNavMenu.classList.remove('active');
            menuToggleButton.setAttribute('aria-expanded', 'false');
        }
    });

    // Optional: Close mobile menu on link click (if links navigate away or to sections)
    const mobileNavLinks = mobileNavMenu.querySelectorAll('a');
    mobileNavLinks.forEach(link => {
        link.addEventListener('click', function() {
              if (mobileNavMenu.classList.contains('active')) {
                mobileNavMenu.classList.remove('active');
                menuToggleButton.setAttribute('aria-expanded', 'false');
            }
        });
    });

    // for theme toggler
    const body = document.body;
    const checkboxDesktop = document.getElementById('checkbox-desktop');
    const checkboxMobile = document.getElementById('checkbox-mobile');

    // load saved theme preference or use system preference
    const savedTheme = document.cookie.split('; ').find(row => row.startsWith('theme='))?.split('=')[1];
    const systemPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = savedTheme || (systemPrefersDark ? 'dark' : 'light');

    // set initial theme and checkbox state
    body.className = initialTheme;
    checkboxDesktop.checked = initialTheme === 'light';
    checkboxMobile.checked = initialTheme === 'light';

    // toggle theme when checkbox is clicked
    checkboxDesktop.addEventListener('change', () => {
      checkboxMobile.checked = checkboxDesktop.checked;
      const newTheme = checkboxDesktop.checked ? 'light' : 'dark';
      body.className = newTheme;
      document.cookie = `theme=${newTheme}; path=/; max-age=31536000`; // expires in 1 year
    });
    // toggle theme when checkbox is clicked
    checkboxMobile.addEventListener('change', () => {
      checkboxDesktop.checked = checkboxMobile.checked;
      const newTheme = checkboxMobile.checked ? 'light' : 'dark';
      body.className = newTheme;
      document.cookie = `theme=${newTheme}; path=/; max-age=31536000`; // expires in 1 year
    });

    // listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
      // only update if no saved preference exists
      // if (!localStorage.getItem('theme')) {
        const newTheme = e.matches ? 'dark' : 'light';
        body.className = newTheme;
        checkboxDesktop.checked = !e.matches;
        document.cookie = `theme=${newTheme}; path=/; max-age=31536000`; // expires in 1 year
      // }
    });

  // Basic Modal Functionality
  const modalTriggers = document.querySelectorAll('.modal-trigger');
  const modals = document.querySelectorAll('.modal');
  const modalCloses = document.querySelectorAll('.modal-close');

  modalTriggers.forEach(trigger => {
      trigger.addEventListener('click', () => {
          const targetModalId = trigger.getAttribute('data-target');
          const modal = document.getElementById(targetModalId);
          if (modal) {
              // Calculate scrollbar width before hiding body scroll
              const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
              document.documentElement.style.setProperty('--scrollbar-width', `${scrollbarWidth}px`);
              modal.classList.add('open');
              document.body.classList.add('modal-open-body-scroll-lock');
          }
      });
  });

  modalCloses.forEach(close => {
      close.addEventListener('click', (e) => {
          e.preventDefault();
          const modal = close.closest('.modal');
          if (modal) {
              modal.classList.remove('open');
              document.body.classList.remove('modal-open-body-scroll-lock');
          }
      });
  });

  modals.forEach(modal => {
      modal.addEventListener('click', (e) => {
          if (e.target === modal) {
              modal.classList.remove('open');
              document.body.classList.remove('modal-open-body-scroll-lock');
          }
      });
  });

  // add escape key listener to close modals
  document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
          const openModal = document.querySelector('.modal.open');
          if (openModal) {
              openModal.classList.remove('open');
              document.body.classList.remove('modal-open-body-scroll-lock');
          }
      }
  });
});
