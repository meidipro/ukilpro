// src/pages/login.ts
import { supabase } from '../supabaseClient';
import { i18n } from '../i18n';

let authMode: 'signIn' | 'signUp' = 'signIn';

async function handleGoogleSignIn() {
    const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: { redirectTo: `${window.location.origin}/app` }
    });
    if (error) {
        console.error("Error signing in with Google:", error);
        alert("Error signing in with Google: " + error.message);
    }
}

async function handleEmailAuth(e: Event) {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const email = (form.elements.namedItem('email') as HTMLInputElement).value;
    const password = (form.elements.namedItem('password') as HTMLInputElement).value;

    if (authMode === 'signUp') {
        const fullName = (form.elements.namedItem('fullName') as HTMLInputElement).value;
        if (!fullName) {
            alert('Please enter your full name.');
            return;
        }

        // --- SIGN UP LOGIC with Full Name ---
        const { data, error } = await supabase.auth.signUp({
            email,
            password,
            options: {
                data: {
                    full_name: fullName // This saves the name to user_metadata
                }
            }
        });
        if (error) {
            alert(error.message);
        } else {
            alert(i18n.t('login_page_confirmEmail'));
            console.log("Sign up successful, user needs to confirm email.", data);
        }
    } else {
        // --- SIGN IN LOGIC (unchanged) ---
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) { alert(error.message); }
    }
}

export function renderLoginPage(container: HTMLElement) {
    const isSignInMode = authMode === 'signIn';
    const title = isSignInMode ? i18n.t('login_page_signInTitle') : i18n.t('login_page_signUpTitle');
    const buttonText = isSignInMode ? i18n.t('login_page_signInButton') : i18n.t('login_page_signUpButton');
    const switchPrompt = isSignInMode ? i18n.t('login_page_askSignUp') : i18n.t('login_page_askSignIn');
    const switchLinkText = isSignInMode ? i18n.t('login_page_linkSignUp') : i18n.t('login_page_linkSignIn');
    
    // --- Dynamically add the Full Name field for Sign Up mode ---
    const fullNameFieldHTML = isSignInMode ? '' : `
      <div class="form-field">
        <label for="fullName">${i18n.t('login_page_fullNameLabel')}</label>
        <input type="text" id="fullName" name="fullName" required autocomplete="name">
      </div>
    `;

    container.innerHTML = `
      <div class="page-container">
        <div class="login-form">
          <h1>${title}</h1>
          <p>${i18n.t('login_page_subtitle')}</p>

          <button id="google-signin-btn" class="google-signin" style="display: flex; align-items: center; justify-content: center;">
            <svg style="margin-right: 12px;" width="20" height="20" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M17.64 9.20455C17.64 8.56636 17.5827 7.95273 17.4764 7.36364H9V10.845H13.8436C13.635 11.97 13.0009 12.9232 12.0477 13.5614V15.8195H14.9564C16.6582 14.2527 17.64 11.9464 17.64 9.20455Z" fill="white"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M9 18C11.43 18 13.4673 17.1941 14.9564 15.8195L12.0477 13.5614C11.2418 14.1014 10.2109 14.4205 9 14.4205C6.96182 14.4205 5.23727 13.0395 4.50545 11.1805H1.51636V13.5095C3.00545 16.2232 5.79409 18 9 18Z" fill="white"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M4.50545 11.1805C4.31636 10.6405 4.20545 10.0695 4.20545 9.47045C4.20545 8.87136 4.31636 8.29909 4.50545 7.76091V5.43182H1.51636C0.952727 6.61955 0.636364 7.97182 0.636364 9.47045C0.636364 10.9691 0.952727 12.3214 1.51636 13.5091L4.50545 11.1805Z" fill="white"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M9 4.52045C10.0214 4.52045 10.9405 4.88727 11.6836 5.59364L15.0218 2.39182C13.4632 0.902727 11.4259 0 9 0C5.79409 0 3.00545 1.77682 1.51636 4.49045L4.50545 6.81954C5.23727 4.96045 6.96182 4.52045 9 4.52045Z" fill="white"></path></svg>
            ${i18n.t('login_page_googleButton')}
          </button>

          <div class="login-divider">${i18n.t('login_page_divider')}</div>

          <form id="email-auth-form" class="email-form">
            ${fullNameFieldHTML}
            <div class="form-field">
                <label for="email">${i18n.t('login_page_emailLabel')}</label>
                <input type="email" id="email" name="email" required autocomplete="email">
            </div>
            <div class="form-field">
                <label for="password">${i18n.t('login_page_passwordLabel')}</label>
                <input type="password" id="password" name="password" required autocomplete="current-password">
            </div>
            <button type="submit">${buttonText}</button>
          </form>

          <p id="switch-auth-mode" class="auth-mode-switch">
            ${switchPrompt} <a href="#">${switchLinkText}</a>
          </p>
        </div>
      </div>
    `;

    document.getElementById('google-signin-btn')?.addEventListener('click', handleGoogleSignIn);
    document.getElementById('email-auth-form')?.addEventListener('submit', handleEmailAuth);
    
    document.getElementById('switch-auth-mode')?.addEventListener('click', (e) => {
        e.preventDefault();
        authMode = (authMode === 'signIn' ? 'signUp' : 'signIn');
        renderLoginPage(container);
    });
}