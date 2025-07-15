
const aboutHTML = `
<div class="about-container animate__animated animate__fadeIn">

  <!-- HERO SECTION -->
  <section class="about-hero">
    <div class="about-hero-content">
      <h1 class="about-title animate__animated animate__fadeInDown">About LegalAI.bd</h1>
      <p class="about-subtitle animate__animated animate__fadeInUp">Democratizing access to Bangladeshi law through the power of artificial intelligence.</p>
    </div>
  </section>

  <!-- MISSION & VISION SECTION -->
  <section class="about-section">
    <div class="about-grid two-cols">
      <div class="about-card animate__animated animate__zoomIn">
        <h3>Our Mission</h3>
        <p>To empower every citizen, student, and legal professional in Bangladesh with instant, clear, and reliable legal information. We bridge the gap between complex legal texts and the people they serve.</p>
      </div>
      <div class="about-card animate__animated animate__zoomIn">
        <h3>Our Vision</h3>
        <p>We envision a future where understanding your rights and legal obligations is no longer a barrier, but a simple conversation away. A future where technology fosters a more just and informed society.</p>
      </div>
    </div>
  </section>

  <!-- HOW IT WORKS INFOGRAPHIC -->
  <section class="about-section">
    <h2 class="section-title animate__animated animate__fadeIn">How It Works</h2>
    <div class="about-grid three-cols infographic">
      <div class="infographic-step animate__animated animate__fadeInUp" style="animation-delay: 0.1s;">
        <div class="infographic-icon">📥</div>
        <h4>1. Ask or Upload</h4>
        <p>Start by asking a question in plain English or Bengali, or upload a legal document for analysis.</p>
      </div>
      <div class="infographic-step animate__animated animate__fadeInUp" style="animation-delay: 0.2s;">
        <div class="infographic-icon">🤖</div>
        <h4>2. AI-Powered Retrieval</h4>
        <p>Our AI searches a curated knowledge base of Bangladeshi laws to find the most relevant information for your query.</p>
      </div>
      <div class="infographic-step animate__animated animate__fadeInUp" style="animation-delay: 0.3s;">
        <div class="infographic-icon">✅</div>
        <h4>3. Get a Clear Answer</h4>
        <p>You receive a structured, easy-to-understand answer, complete with legal citations and actionable steps.</p>
      </div>
    </div>
  </section>

  <!-- TEAM SECTION -->
  <section class="about-section team-section">
    <h2 class="section-title">Meet Our Team</h2>
    <div class="about-grid two-cols">
      <div class="team-card animate__animated animate__fadeInLeft">
        <img src="https://avatars.githubusercontent.com/u/86288624?v=4" alt="Mehide Hasan" class="team-photo">
        <h3>Mehide Hasan</h3>
        <p class="team-role">CO-Founder, Full-Stack Developer & Project Managgreer</p>
        <p>Designed, built, and managed the entire application — from front-end UI/UX to back-end logic and deployment.</p>
      </div>
      <div class="team-card animate__animated animate__fadeInRight">
        <img src="https://media.licdn.com/dms/image/D5603AQG859G9o21jHw/profile-displayphoto-shrink_400_400/0/1718563998655?e=1727308800&v=beta&t=3E1zHnYM01u6eBSo_t2j5qV2j5iYFkZ3e2c5Y5E3E2c" alt="Tajuddin Ahmed" class="team-photo">
        <h3>Tajuddin Ahmed</h3>
        <p class="team-role">CO-Founder, AI Knowledge Curator & Tester</p>
        <p>Helped gather Bangladeshi legal knowledge, tested features, and ensured that the AI response quality meets public needs.</p>
      </div>
    </div>
  </section>

  <!-- FAQ SECTION -->
  <section class="about-section faq-section">
    <h2 class="section-title">Frequently Asked Questions</h2>
    <div class="faq-container">
      <div class="faq-item">
        <button class="faq-question">Is LegalAI.bd free to use?</button>
        <div class="faq-answer">
          <p>Yes, LegalAI.bd offers a generous free tier for guest users and registered users. For heavy usage or advanced features, we may introduce premium plans in the future to cover operational costs.</p>
        </div>
      </div>
      <div class="faq-item">
        <button class="faq-question">Is this a substitute for a real lawyer?</button>
        <div class="faq-answer">
          <p><strong>No.</strong> The information provided by LegalAI.bd is for informational and educational purposes only. It is not legal advice and should not be used as a substitute for consultation with a qualified legal professional. Always consult a lawyer for serious legal matters.</p>
        </div>
      </div>
      <div class="faq-item">
        <button class="faq-question">How up-to-date is the legal information?</button>
        <div class="faq-answer">
          <p>We strive to keep our knowledge base as current as possible by incorporating the latest laws and amendments from official government sources. However, there may be a delay between the official publication of a new law and its inclusion in our database.</p>
        </div>
      </div>
    </div>
  </section>

  

  <!-- FINAL CTA SECTION -->
  <section class="about-section final-cta-section">
    <h2 class="section-title">Ready to Explore?</h2>
    <p>Get instant answers to your legal questions and simplify complex laws.</p>
    <a href="/app" class="cta-button cta-primary" data-link>Try LegalAI Now</a>
  </section>

</div>
`;

export function renderAboutPage(container: HTMLElement) {
    const animateCSSLinkId = 'animate-css-cdn';
    if (!document.getElementById(animateCSSLinkId)) {
        const link = document.createElement('link');
        link.id = animateCSSLinkId;
        link.rel = 'stylesheet';
        link.href = 'https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css';
        document.head.appendChild(link);
    }

    container.innerHTML = aboutHTML;

    // Add event listeners for the FAQ accordion
    const faqItems = container.querySelectorAll('.faq-item');
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        const answer = item.querySelector('.faq-answer');
        question?.addEventListener('click', () => {
            const isVisible = answer?.classList.contains('visible');
            // Close all other answers
            faqItems.forEach(otherItem => {
                otherItem.querySelector('.faq-answer')?.classList.remove('visible');
                otherItem.querySelector('.faq-question')?.classList.remove('active');
            });
            // Toggle the clicked one
            if (!isVisible) {
                answer?.classList.add('visible');
                question?.classList.add('active');
            }
        });
    });
}
