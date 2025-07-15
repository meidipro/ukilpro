// src/pages/user-profile.ts
import { auth } from '../auth';
import { supabase } from '../supabaseClient';
import { i18n } from '../i18n'; // <-- IMPORT i18n

async function handleProfileUpdate(e: Event) {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const formData = new FormData(form);
    const fullName = formData.get('fullName') as string;
    const user = auth.getSession()?.user;

    if (!user) {
        alert(i18n.t('profile_alert_notLoggedInUpdate')); // <-- TRANSLATED
        return;
    }

    const { data, error } = await supabase.auth.updateUser({
        data: { full_name: fullName }
    });

    if (error) {
        alert(`${i18n.t('profile_alert_error')} ${error.message}`); // <-- TRANSLATED
    } else {
        alert(i18n.t('profile_alert_success')); // <-- TRANSLATED
        console.log("Updated user data:", data.user);
    }
}


export function renderUserProfilePage(container: HTMLElement) {
    const session = auth.getSession();

    if (!session) {
        container.innerHTML = `<div class="page-container"><p>${i18n.t('profile_alert_notLoggedInView')}</p></div>`; // <-- TRANSLATED
        return;
    }

    const fullName = session.user.user_metadata?.full_name || '';
    
    // --- UPDATED HTML with i18n keys ---
    container.innerHTML = `
      <div class="page-container">
        <form class="login-form" id="profile-form">
          <h1>${i18n.t('profile_title')}</h1>
          <p>${i18n.t('profile_subtitle')}</p>
          
          <div>
            <label for="email" style="display: block; text-align: left; margin-bottom: 4px;">${i18n.t('profile_email')}</label>
            <input type="email" id="email" value="${session.user.email}" disabled>
          </div>
          
          <div>
            <label for="fullName" style="display: block; text-align: left; margin-bottom: 4px;">${i18n.t('profile_fullName')}</label>
            <input type="text" id="fullName" name="fullName" value="${fullName}" placeholder="${i18n.t('profile_fullNamePlaceholder')}">
          </div>

          <button type="submit">${i18n.t('profile_updateButton')}</button>
        </form>
      </div>
    `;

    const profileForm = document.getElementById('profile-form');
    profileForm?.addEventListener('submit', handleProfileUpdate);
}