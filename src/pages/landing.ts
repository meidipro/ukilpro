// src/pages/landing.ts
import { auth } from '../auth'; // <-- CORRECTED PATH
import { i18n } from '../i18n';   // <-- CORRECTED PATH

declare const particlesJS: any;

// --- Main render function for the landing page ---
export function renderLandingPage(container: HTMLElement) {
    // --- NEW: Check the user's session state ---
    const session = auth.getSession();

    // --- NEW: Dynamically generate the Call-to-Action buttons ---
    let ctaButtonsHTML = '';
    if (session) {
        // User is LOGGED IN: Show a single button to go to the app.
        ctaButtonsHTML = `
            <a href="/app" class="cta-button cta-primary" data-link>${i18n.t('landing_cta_goToApp')}</a>
        `;
    } else {
        // User is LOGGED OUT: Show the original two buttons.
        ctaButtonsHTML = `
            <a href="/app" class="cta-button cta-secondary" data-link>${i18n.t('landing_cta_guest')}</a>
            <a href="/login" class="cta-button cta-primary" data-link>${i18n.t('landing_cta_signup')}</a>
        `;
    }

    // --- Build the final HTML using the dynamic buttons ---
    const landingHTML = `
      <div class="landing-page-wrapper">
        <section class="hero-section">
          <div id="particles-js"></div> 
          <div class="hero-content">
            <h1>${i18n.t('landing_heroTitle_1')} <span class="highlight">${i18n.t('landing_heroTitle_2')}</span></h1>
            <p>${i18n.t('landing_heroSubtitle')}</p>
            <div class="cta-buttons">
              ${ctaButtonsHTML}
            </div>
          </div>
        </section>

        <section id="features" class="features-section fade-in-section">
          <div class="section-header">
            <h2>${i18n.t('landing_featuresTitle')}</h2>
            <p>${i18n.t('landing_featuresSubtitle')}</p>
          </div>
          <div class="features-grid">
            <div class="feature-card">
              <svg class="feature-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-4.67c.625-.93.928-2.036.928-3.185V11a3 3 0 00-3-3H9.375a3 3 0 00-3 3v1.375c0 1.15.303 2.255.927 3.185z" /></svg>
              <h3>${i18n.t('landing_feature1_title')}</h3>
              <p>${i18n.t('landing_feature1_text')}</p>
            </div>
            <div class="feature-card">
              <svg class="feature-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" /></svg>
              <h3>${i18n.t('landing_feature2_title')}</h3>
              <p>${i18n.t('landing_feature2_text')}</p>
            </div>
            <div class="feature-card">
              <svg class="feature-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 21h16.5M4.5 3h15M5.25 3v18M18.75 3v18M9 21V7.5M15 21V7.5M3 12h18M3 15h18M3 9h18" /></svg>
              <h3>${i18n.t('landing_feature3_title')}</h3>
              <p>${i18n.t('landing_feature3_text')}</p>
            </div>
          </div>
        </section>

        <footer class="footer">
          <p>© ${new Date().getFullYear()} LegalAI.bd - All Rights Reserved.</p>
          <div class="footer-links">
            <a href="/about" data-link>${i18n.t('footer_about')}</a>
            <a href="#">${i18n.t('footer_privacy')}</a>
            <a href="#">${i18n.t('footer_terms')}</a>
          </div>
        </footer>
      </div>
    `;

    container.innerHTML = landingHTML;
    initParticles(); 
    setupFadeInAnimations();
}


// --- These helper functions remain unchanged ---
function initParticles() {
  if (document.getElementById('particles-js')) {
    particlesJS('particles-js', { particles: { number: { value: 60, density: { enable: true, value_area: 800 } }, color: { value: "#a78bfa" }, shape: { type: "circle" }, opacity: { value: 0.4, random: true }, size: { value: 3, random: true }, line_linked: { enable: true, distance: 150, color: "#6366f1", opacity: 0.2, width: 1 }, move: { enable: true, speed: 1.5, direction: "none", random: true, straight: false, out_mode: "out" } }, interactivity: { detect_on: "canvas", events: { onhover: { enable: true, mode: "grab" }, onclick: { enable: false }, resize: true }, modes: { grab: { distance: 140, line_linked: { opacity: 0.5 } } } }, retina_detect: true });
  }
}

function setupFadeInAnimations() {
  const sections = document.querySelectorAll('.fade-in-section');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  sections.forEach(section => { observer.observe(section); });
}