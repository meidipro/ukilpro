import { marked } from 'marked';
import { supabase } from '../supabaseClient';
import { auth } from '../auth';
import { i18n } from '../i18n';
import Groq from "groq-sdk";

import mermaid from 'mermaid';

// --- Types matching your DB schema ---
type Sender = 'user' | 'ai';
interface Message { id?: number; sender: Sender; content: string; }
interface Chat { id: string; title: string; messages: Message[]; dify_conversation_id?: string; has_document?: boolean; dify_file_ids?: string[]; }
interface AppState { chats: Chat[]; activeChatId: string | null; }

interface SpeechRecognitionEvent extends Event { results: SpeechRecognitionResultList; }
interface SpeechRecognitionErrorEvent extends Event { error: string; }
declare var webkitSpeechRecognition: any;
declare var SpeechRecognition: any;
declare global {
    interface Window {
        SpeechRecognition?: typeof SpeechRecognition;
        webkitSpeechRecognition?: typeof webkitSpeechRecognition;
    }
}

export async function renderAppPage(container: HTMLElement) {
    const DIFY_REVIEWER_API_KEY = import.meta.env.VITE_DIFY_REVIEWER_API_KEY;
    const DIFY_GENERAL_API_KEY = import.meta.env.VITE_DIFY_GENERAL_API_KEY;
    const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_KEY;
    const GUEST_STORAGE_KEY = 'legalAI.guestChats';
    const GUEST_USER_ID_KEY = 'legalAI.guestUserId';

    const groq = new Groq({ apiKey: GROQ_API_KEY, dangerouslyAllowBrowser: true });

    let appState: AppState;
    const session = auth.getSession();
    const isGuestMode = session === null;

    const SUGGESTED_QUERIES = ['app_query_1', 'app_query_2', 'app_query_3', 'app_query_4'];

    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = SpeechRecognitionAPI ? new SpeechRecognitionAPI() : null;
    const synthesis = window.speechSynthesis;
    let isListening = false;

    function getOrCreateGuestUserId(): string {
        let guestId = localStorage.getItem(GUEST_USER_ID_KEY);
        if (!guestId) {
            guestId = `guest_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
            localStorage.setItem(GUEST_USER_ID_KEY, guestId);
        }
        return guestId;
    }
    const userIdentifier = session?.user?.id || getOrCreateGuestUserId();

    // --- HTML Structure ---
    container.innerHTML = `
      <div class="app-layout">
          <aside class="sidebar">
              <div class="sidebar-top">
                <button class="new-chat-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                    ${i18n.t('app_newChat')}
                </button>
                <div class="sidebar-role-selector">
                    <label for="role-selector">${i18n.t('app_iAmA')}</label>
                    <select id="role-selector">
                        <option value="General Public" selected>${i18n.t('app_role_general')}</option>
                        <option value="Law Student">${i18n.t('app_role_student')}</option>
                        <option value="Legal Professional">${i18n.t('app_role_professional')}</option>
                    </select>
                </div>
              </div>
              <div class="conversation-list"><h2>${i18n.t('app_history')}</h2></div>
              <div class="sidebar-footer">
                  <div id="dark-mode-toggle">
                       <svg class="icon" id="theme-icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path></svg>
                       <span id="theme-text">${document.body.classList.contains('dark-mode') ? i18n.t('app_lightMode') : i18n.t('app_darkMode')}</span>
                  </div>
                  <div id="sidebar-lang-switcher" class="language-switcher-sidebar">
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
                      <span class="lang-en ${i18n.getLanguage() === 'en' ? 'lang-active' : ''}">EN</span>
                      <span>/</span>
                      <span class="lang-bn ${i18n.getLanguage() === 'bn' ? 'lang-active' : ''}">বাং</span>
                  </div>
                  <div id="user-profile-link" class="user-profile-link"></div>
              </div>
          </aside>
          <main class="main-content">
              <div id="chat-window"></div>
              <div class="message-form-container">
                  <form id="message-form">
                      <button type="button" id="upload-doc-btn" class="upload-btn" title="Analyze a document">
                          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>
                      </button>
                      <input type="file" id="doc-file-input" style="display:none" accept=".pdf,.doc,.docx,.txt,.rtf,.odt,.jpg,.jpeg,.png,.heic,.webp,.csv,.xlsx,.xls,.ppt,.pptx,.zip,.rar,.7z,.tar,.gz,.json,.xml,.eml,.msg,.md,.html,.htm,.epub,.mobi,.azw,.azw3,.fb2,.djvu,.cbz,.cbr,.mp3,.wav,.mp4,.mov,.avi,.mkv,.flv,.wmv,.ogg,.aac,.flac,.m4a,.webm,.ts,.m4v,.3gp,.3g2,.m2ts,.mts,.vob,.dat,.iso,.swf,.svg,.psd,.ai,.indd,.xd,.sketch,.fig,.blend,.dwg,.dxf,.stl,.obj,.fbx,.gltf,.glb,.3ds,.max,.c4d,.lwo,.lws,.ma,.mb,.ase,.aseprite,.spr,.sai,.clip,.kra,.ora,.psb,.tif,.tiff,.bmp,.ico,.icns,.cur,.ani,.exr,.hdr,.dds,.tga,.pal,.act,.thm,.yuv,.cin,.dpx,.rle,.sgi,.bw,.rgb,.rgba,.int,.inta,.sid,.pcx,.pict,.pct,.pic,.mac,.mag,.img,.sun,.ras,.xpm,.xbm,.ppm,.pgm,.pbm,.pnm,.pam,.j2k,.jp2,.jpf,.jpx,.jpm,.mj2,.svgz,.webp,.avif,.apng,.jxl,.heif,.heic,.pdf,.djvu,.xps,.oxps,.cbz,.cbr,.cbt,.cba,.cb7,.zip,.rar,.7z,.tar,.gz,.bz2,.xz,.lz,.lzma,.z,.tz,.tbz,.tgz,.txz,.tlz,.lz4,.lzo,.sz,.zst,.cab,.arj,.ace,.uue,.bz,.bzip2,.gzip,.lz,.lzma,.lzo,.rar,.xz,.z,.zip,.001,.7z,.ace,.alz,.apk,.arj,.bin,.bz2,.cab,.cpio,.deb,.dmg,.egg,.gz,.hqx,.img,.iso,.jar,.lzh,.lzma,.lzo,.msi,.pkg,.rar,.rpm,.sea,.sit,.tar,.tbz2,.tgz,.tlz,.txz,.war,.wim,.xar,.xz,.z,.zip,.zipx" />
                      <button type="button" id="mic-button" class="mic-btn" title="Ask with voice">
                          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line></svg>
                      </button>
                      <input type="text" id="message-input" placeholder="${i18n.t('app_askAnything')}" autocomplete="off" required>
                      <button type="submit" id="send-button">
                          <svg class="icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                      </button>
                  </form>
              </div>
          </main>
          <div id="overlay"></div>
      </div>`;

    const uploadDocBtn = document.getElementById('upload-doc-btn') as HTMLButtonElement;
    const docFileInput = document.getElementById('doc-file-input') as HTMLInputElement;
    const sidebar = document.querySelector('.sidebar') as HTMLElement;
    const overlay = document.getElementById('overlay') as HTMLDivElement;
    const chatWindow = document.getElementById('chat-window') as HTMLDivElement;
    const messageForm = document.getElementById('message-form') as HTMLFormElement;
    const messageInput = document.getElementById('message-input') as HTMLInputElement;
    const newChatBtn = document.querySelector('.new-chat-btn') as HTMLButtonElement;
    const conversationList = document.querySelector('.conversation-list') as HTMLDivElement;
    const darkModeToggle = document.getElementById('dark-mode-toggle') as HTMLDivElement;
    const themeText = document.getElementById('theme-text') as HTMLSpanElement;
    const userProfileLink = document.getElementById('user-profile-link') as HTMLDivElement;
    const micButton = document.getElementById('mic-button') as HTMLButtonElement;
    const sidebarLangSwitcher = document.getElementById('sidebar-lang-switcher');

    function speakText(text: string) {
        if (synthesis.speaking) synthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        const voices = synthesis.getVoices();
        const langCode = i18n.getLanguage();
        const preferredVoice = voices.find(voice => voice.lang.startsWith(langCode) && voice.name.includes('Google'));
        utterance.voice = preferredVoice || voices.find(voice => voice.lang.startsWith(langCode)) || voices[0];
        const lastMessageAvatar = chatWindow.querySelector('.message-wrapper:last-child .ai-avatar');
        utterance.onstart = () => { lastMessageAvatar?.classList.add('is-speaking'); };
        utterance.onend = () => { lastMessageAvatar?.classList.remove('is-speaking'); };
        utterance.onerror = () => { lastMessageAvatar?.classList.remove('is-speaking'); };
        utterance.rate = 1;
        utterance.pitch = 1;
        synthesis.speak(utterance);
    }

    function getActiveChat(): Chat | undefined {
        if (!appState || !appState.activeChatId) return undefined;
        return appState.chats.find(c => c.id === appState.activeChatId);
    }

    function renderSidebar() {
        if (!conversationList) return;
        conversationList.innerHTML = `<h2>${i18n.t('app_history')}</h2>`;
        (appState.chats || []).forEach(chat => {
            const convoItem = document.createElement('div');
            convoItem.className = 'conversation-item';
            if (chat.id === appState.activeChatId) convoItem.classList.add('active');

            const titleArea = document.createElement('div');
            titleArea.className = 'conversation-title-area';
            titleArea.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 21 1.65-3.8a9 9 0 1 1 3.4 2.9l-5.05.9"></path></svg><span>${chat.title}</span>`;
            titleArea.addEventListener('click', () => setActiveChat(chat.id));
            convoItem.appendChild(titleArea);

            const actionsMenu = document.createElement('div');
            actionsMenu.className = 'conversation-actions';
            actionsMenu.innerHTML = `
                <button class="action-btn rename-btn" title="Rename">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                </button>
                <button class="action-btn share-btn" title="Share">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"></path><polyline points="16 6 12 2 8 6"></polyline><line x1="12" y1="2" x2="12" y2="15"></line></svg>
                </button>
                <button class="action-btn delete-btn" title="Delete">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            `;
            actionsMenu.querySelector('.rename-btn')?.addEventListener('click', (e) => { e.stopPropagation(); renameChat(chat.id); });
            actionsMenu.querySelector('.share-btn')?.addEventListener('click', (e) => { e.stopPropagation(); shareChat(chat.id, e.currentTarget as HTMLButtonElement); });
            actionsMenu.querySelector('.delete-btn')?.addEventListener('click', (e) => { e.stopPropagation(); deleteChat(chat.id); });

            convoItem.appendChild(actionsMenu);
            conversationList.appendChild(convoItem);
        });
    }

    function renderChatWindow() {
        if (!chatWindow) return;
        const activeChat = getActiveChat();
        const showWelcomeScreen = !activeChat || (activeChat.messages.length === 1 && activeChat.messages[0].sender === 'ai');
        if (showWelcomeScreen) {
            const suggestedQueriesHTML = SUGGESTED_QUERIES.map(key => `<div class="suggested-query-item">${i18n.t(key as any)}</div>`).join('');
            chatWindow.innerHTML = `
                <div class="empty-chat-container"> 
                    <div class="empty-chat-logo">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v17.25m0 0c-1.472 0-2.882.265-4.185.75M12 20.25c1.472 0 2.882.265 4.185.75M18.75 4.97A48.416 48.416 0 0012 4.5c-2.291 0-4.545.16-6.75.47m13.5 0c1.01.143 2.01.317 3 .52m-3-.52l2.62 10.726c.122.499-.106 1.028-.589 1.202a5.988 5.988 0 01-2.153.34c-1.325 0-2.59-.523-3.536-1.465l-2.62-2.62m5.156 0l-2.62 2.62m-5.156 0l-2.62-2.62m6.75-10.726C12 4.5 11.25 4.5 10.5 4.5c-1.01 0-2.01.143-3 .52m3-.52l-2.62 10.726" /></svg> 
                    </div> 
                    <h2>${i18n.t('app_emptyChatTitle')}</h2>
                    <div class="suggested-queries-container">
                        ${suggestedQueriesHTML}
                    </div>
                </div>`;
        } else {
            chatWindow.innerHTML = '';
            activeChat.messages.forEach(msg => displayMessage(msg.content, msg.sender));
        }
    }

    function displayMessage(text: string, sender: Sender): HTMLDivElement {
        if (!chatWindow) return document.createElement('div');

        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message-wrapper ${sender}`;
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        if (sender === 'user') {
            avatar.classList.add('user-avatar');
            const user = auth.getSession()?.user;
            avatar.textContent = user?.email?.charAt(0).toUpperCase() || 'G';
        } else {
            avatar.classList.add('ai-avatar');
            avatar.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v17.25m0 0c-1.472 0-2.882.265-4.185.75M12 20.25c1.472 0 2.882.265 4.185.75M18.75 4.97A48.416 48.416 0 0012 4.5c-2.291 0-4.545.16-6.75.47m13.5 0c1.01.143 2.01.317 3 .52m-3-.52l2.62 10.726c.122.499-.106 1.028-.589 1.202a5.988 5.988 0 01-2.153.34c-1.325 0-2.59-.523-3.536-1.465l-2.62-2.62m5.156 0l-2.62 2.62m-5.156 0l-2.62-2.62m6.75-10.726C12 4.5 11.25 4.5 10.5 4.5c-1.01 0-2.01.143-3 .52m3-.52l-2.62 10.726" /></svg>`;
        }
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        if (sender === 'ai') {
            const senderName = document.createElement('div');
            senderName.className = 'sender-name';
            senderName.textContent = i18n.t('app_aiSenderName');
            messageContent.appendChild(senderName);
        }
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';

        // --- NEW: Logic to parse follow-up questions ---
        let mainContent = text;
        let followUpQuestions: string[] = [];
        const followUpRegex = /<h4>Follow-up Suggestions:<\/h4>\s*<ol>(.*?)<\/ol>/s;
        const match = text.match(followUpRegex);

        if (match) {
            mainContent = text.replace(followUpRegex, '').trim();
            const listItemsRegex = /<li>(.*?)<\/li>/g;
            let itemMatch;
            while ((itemMatch = listItemsRegex.exec(match[1])) !== null) {
                followUpQuestions.push(itemMatch[1]);
            }
        }
        // --- END of new logic ---

        if (sender === 'ai') {
            const parsed = marked.parse(mainContent, { gfm: true });
            if (parsed instanceof Promise) {
                parsed.then(html => {
                    messageBubble.innerHTML = html;
                    messageBubble.querySelectorAll('.language-mermaid').forEach(async (el) => {
                        if (el instanceof HTMLElement) {
                            try {
                                const { svg } = await mermaid.render(el.id, el.textContent || '');
                                el.innerHTML = svg;
                            } catch (e) {
                                el.innerHTML = `Error rendering Mermaid diagram: ${e instanceof Error ? e.message : e}`;
                            }
                        }
                    });
                });
            } else {
                messageBubble.innerHTML = parsed;
                messageBubble.querySelectorAll('.language-mermaid').forEach(async (el) => {
                    if (el instanceof HTMLElement) {
                        const uniqueId = `mermaid-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
                        el.id = uniqueId;
                        try {
                            const { svg } = await mermaid.render(uniqueId, el.textContent || '');
                            el.innerHTML = svg;
                        } catch (e) {
                            el.innerHTML = `Error rendering Mermaid diagram: ${e instanceof Error ? e.message : e}`;
                        }
                    }
                });
            }
        } else {
            messageBubble.innerText = mainContent;
        }
        messageContent.appendChild(messageBubble);

        // --- Controls for AI messages ---
        if (sender === 'ai' && text.trim() !== "" && !text.includes('thinking')) {
            const controlsWrapper = document.createElement('div');
            controlsWrapper.className = 'message-controls';

            const feedbackControls = document.createElement('div');
            feedbackControls.className = 'feedback-controls';
            const thumbUp = document.createElement('button');
            thumbUp.className = 'feedback-btn';
            thumbUp.innerHTML = '👍';
            thumbUp.title = 'Good response';
            const thumbDown = document.createElement('button');
            thumbDown.className = 'feedback-btn';
            thumbDown.innerHTML = '👎';
            thumbDown.title = 'Bad response';

            thumbUp.addEventListener('click', () => {
                sendFeedback(getActiveChat()?.id ?? '', text, 'good');
                thumbUp.disabled = true;
                thumbDown.disabled = true;
                thumbUp.classList.add('selected');
            });

            thumbDown.addEventListener('click', () => {
                sendFeedback(getActiveChat()?.id ?? '', text, 'bad');
                thumbUp.disabled = true;
                thumbDown.disabled = true;
                thumbDown.classList.add('selected');
            });

            feedbackControls.appendChild(thumbUp);
            feedbackControls.appendChild(thumbDown);

            const speakButton = document.createElement('button');
            speakButton.className = 'speak-btn';
            speakButton.title = 'Read this message aloud';
            speakButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>`;
            speakButton.addEventListener('click', (e) => {
                e.stopPropagation();
                speakText(mainContent); // Speak only the main content
            });

            controlsWrapper.appendChild(feedbackControls);
            controlsWrapper.appendChild(speakButton);
            messageContent.appendChild(controlsWrapper);
        }

        if (sender === 'user') {
            messageWrapper.appendChild(messageContent);
            messageWrapper.appendChild(avatar);
        } else {
            messageWrapper.appendChild(avatar);
            messageWrapper.appendChild(messageContent);
        }
        chatWindow.appendChild(messageWrapper);

        // --- NEW: Render follow-up questions ---
        if (followUpQuestions.length > 0) {
            const followUpContainer = document.createElement('div');
            followUpContainer.className = 'follow-up-container';
            followUpQuestions.forEach(question => {
                const button = document.createElement('button');
                button.className = 'follow-up-question';
                button.textContent = question;
                followUpContainer.appendChild(button);
            });
            chatWindow.appendChild(followUpContainer);
        }
        // --- END of new rendering ---

        chatWindow.scrollTop = chatWindow.scrollHeight;

        return messageWrapper;
    }

    async function setActiveChat(id: string | null) {
        if (!id) { appState.activeChatId = null; renderSidebar(); renderChatWindow(); return; }
        if (appState.activeChatId === id) return;
        appState.activeChatId = id;
        const activeChat = getActiveChat();
        if (activeChat && activeChat.messages.length === 0 && !isGuestMode) {
            const { data, error } = await supabase.from('messages').select('id, content, sender').eq('chat_id', id).order('created_at', { ascending: true });
            if (error) console.error('Error fetching messages:', error);
            else if (data) activeChat.messages = data as Message[];
        }
        renderSidebar();
        renderChatWindow();
        if (window.innerWidth <= 900) { sidebar.classList.remove('is-open'); overlay.classList.remove('is-open'); }
    }

    // --- FINAL, CORRECTED loadState FUNCTION ---

    async function loadState() {
        if (isGuestMode) {
            const savedState = localStorage.getItem(GUEST_STORAGE_KEY);
            appState = savedState ? JSON.parse(savedState) : { chats: [], activeChatId: null };
        } else {
            // VVV THIS IS THE FIX VVV
            const { data, error } = await supabase
                .from('chats')
                .select('id, title, has_document') // <<< FIX: Request the new column
                .order('created_at', { ascending: false });
            // ^^^ THIS IS THE FIX ^^^

            if (error) {
                console.error("Error fetching chats:", error);
                appState = { chats: [], activeChatId: null };
                return;
            }

            // The spread operator `...c` will now correctly include the `has_document` property
            appState = { chats: data.map((c: any) => ({ ...c, messages: [] })), activeChatId: null };
        }
    }

    async function createNewChat() {
        const initialMessage = { sender: 'ai' as Sender, content: i18n.t('app_initialGreeting') };
        if (isGuestMode) {
            const newChat: Chat = { id: `guest_${Date.now()}`, title: i18n.t('app_newChat'), messages: [initialMessage] };
            appState.chats.unshift(newChat);
            await setActiveChat(newChat.id);
            localStorage.setItem(GUEST_STORAGE_KEY, JSON.stringify(appState));
        } else {
            const { data: { user } } = await supabase.auth.getUser();
            if (!user) return;
            const { data, error } = await supabase.from('chats').insert({ user_id: user.id, title: i18n.t('app_newChat') }).select().single();
            if (error) { console.error("Error creating chat:", error); return; }
            const newChat: Chat = { ...data, messages: [initialMessage] };
            appState.chats.unshift(newChat);
            await setActiveChat(newChat.id);
        }
    }

    async function addMessageToActiveChat(message: Message, difyId?: string) {
        const activeChat = getActiveChat();
        if (!activeChat) return;
        activeChat.messages.push(message);
        if (activeChat.messages.length === 2 && message.sender === 'user') {
            const newTitle = message.content.substring(0, 30) + (message.content.length > 30 ? '...' : '');
            activeChat.title = newTitle;
            if (!isGuestMode) await supabase.from('chats').update({ title: newTitle }).eq('id', activeChat.id);
        }
        if (difyId) activeChat.dify_conversation_id = difyId;
        if (isGuestMode) {
            localStorage.setItem(GUEST_STORAGE_KEY, JSON.stringify(appState));
        } else {
            const { data: { user } } = await supabase.auth.getUser();
            if (user) {
                await supabase.from('messages').insert({
                    chat_id: activeChat.id, user_id: user.id,
                    sender: message.sender, content: message.content
                });
            }
        }
        renderSidebar();
        renderChatWindow();
    }

    async function renameChat(id: string) {
        const chat = appState.chats.find(c => c.id === id);
        if (!chat) return;
        const newTitle = prompt(i18n.t('app_renameTitlePrompt'), chat.title);
        if (newTitle && newTitle.trim() !== "") {
            chat.title = newTitle.trim();
            if (isGuestMode) localStorage.setItem(GUEST_STORAGE_KEY, JSON.stringify(appState));
            else await supabase.from('chats').update({ title: newTitle.trim() }).eq('id', id);
            renderSidebar();
        }
    }

    async function deleteChat(id: string) {
        if (!confirm(i18n.t('app_deleteConfirm'))) return;
        const index = appState.chats.findIndex(c => c.id === id);
        if (index > -1) {
            appState.chats.splice(index, 1);
            if (isGuestMode) localStorage.setItem(GUEST_STORAGE_KEY, JSON.stringify(appState));
            else await supabase.from('chats').delete().eq('id', id);
            if (appState.activeChatId === id) {
                const nextId = appState.chats.length > 0 ? appState.chats[0].id : null;
                if (nextId) await setActiveChat(nextId);
                else await createNewChat();
            } else {
                renderSidebar();
            }
        }
    }

    function shareChat(id: string, button: HTMLButtonElement) {
        const chat = appState.chats.find(c => c.id === id);
        if (!chat) return;
        const formattedChat = `Chat: ${chat.title}\n\n${chat.messages.map(msg => `${msg.sender === 'user' ? 'You' : 'LegalAI'}:\n${msg.content}`).join('\n\n')}`;
        navigator.clipboard.writeText(formattedChat).then(() => {
            const originalContent = button.innerHTML;
            button.innerHTML = '✅';
            button.disabled = true;
            setTimeout(() => { button.innerHTML = originalContent; button.disabled = false; }, 2000);
        }).catch(err => { console.error('Failed to copy chat:', err); alert('Failed to copy chat.'); });
    }

    async function sendFeedback(chatId: string, msgContent: string, rating: 'good' | 'bad') {
        const { error } = await supabase
            .from('message_feedback')
            .insert({
                chat_id: chatId,
                message_content: msgContent,
                rating: rating
            });
        if (error) {
            console.error('Error saving feedback:', error);
        }
    }

    // --- Updated handleFormSubmit with AI switching and all previous features ---

    async function handleFormSubmit() {
        const userInput = messageInput.value.trim();
        if (!userInput) return;
        if (!appState.activeChatId) await createNewChat();

        const activeChat = getActiveChat();
        if (!activeChat) return;

        // --- FIX: Link #1 - Select the correct AI ---
        const isDocumentChat = activeChat.has_document === true;
        const API_KEY = isDocumentChat ? DIFY_REVIEWER_API_KEY : DIFY_GENERAL_API_KEY;

        if (!API_KEY) {
            const errorMsg = "Error: The AI for this mode is not configured. Please check API keys.";
            console.error(errorMsg);
            displayMessage(errorMsg, 'ai');
            return;
        }

        await addMessageToActiveChat({ sender: 'user', content: userInput });
        messageInput.value = '';
        const tempMessageWrapper = displayMessage(i18n.t('app_thinking'), 'ai');

        try {
            // --- FIX: Link #2 - Attach the file ID ---
            let filesToAttach: { type: string; upload_file_id: string; }[] = [];
            if (activeChat.dify_file_ids && activeChat.dify_file_ids.length > 0) {
                filesToAttach = activeChat.dify_file_ids.map(id => ({
                    type: "file",
                    upload_file_id: id,
                }));

            }

            const { fullResponse, finalDifyId } = await sendQueryToDify(
                userInput,
                filesToAttach,
                activeChat.dify_conversation_id || "",
                API_KEY,
                tempMessageWrapper
            );
            await addMessageToActiveChat({ sender: 'ai', content: fullResponse }, finalDifyId);

            // Fallback to Groq if Dify provides no answer
            if (!fullResponse.trim()) {
                const groqResponse = await groq.chat.completions.create({
                    messages: [{ role: "user", content: userInput }],
                    model: "llama3-8b-8192"
                });
                const groqAnswer = groqResponse.choices[0]?.message?.content || "";
                if (groqAnswer) {
                    await addMessageToActiveChat({ sender: 'ai', content: groqAnswer });
                } else {
                    throw new Error("Both Dify and Groq failed to provide a response.");
                }
            }
        } catch (error) {
            tempMessageWrapper?.remove();
            const errorMessage = `${i18n.t('app_error')} ${error instanceof Error ? error.message : 'Unknown error'}`;
            await addMessageToActiveChat({ sender: 'ai', content: errorMessage });
            speakText(errorMessage);
        }
    }

    async function sendQueryToDify(query: string, files: Array<{ type: string; upload_file_id: string; }>, conversationId: string, apiKey: string, tempMessageWrapper: HTMLDivElement) {
        const requestBody = JSON.stringify({
            inputs: {
                "LANGUAGE": i18n.getLanguage() === 'bn' ? 'Bengali' : 'English',
                "USER_ROLE": (document.getElementById('role-selector') as HTMLSelectElement).value,
                "QUERY": query.substring(0, 45) // Truncate for the input form
            },
            query: query, // Send the full query here
            user: userIdentifier,
            conversation_id: conversationId,
            response_mode: 'streaming',
            files: files
        });

        console.log("Dify chat API request body:", requestBody);

        const response = await fetch('https://api.dify.ai/v1/chat-messages', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
            body: requestBody
        });

        if (!response.ok) {
            const errorBody = await response.text();
            throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorBody}`);
        }

        tempMessageWrapper.remove();

        let fullResponse = "";
        let finalDifyId = conversationId;
        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        const aiMessageWrapper = displayMessage("", 'ai');
        const aiMessageBubble = aiMessageWrapper.querySelector('.message-bubble') as HTMLDivElement;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n').filter(line => line.startsWith('data: '));

            for (const line of lines) {
                try {
                    const jsonStr = line.substring(6);
                    if (jsonStr.trim() === '[DONE]') continue;
                    const parsedJson = JSON.parse(jsonStr);
                    if (parsedJson.conversation_id) finalDifyId = parsedJson.conversation_id;
                    if (parsedJson.event === 'agent_message' || parsedJson.event === 'message') {
                        fullResponse += parsedJson.answer;
                        const parsedMarkdown = marked.parse(fullResponse, { gfm: true });
                        if (parsedMarkdown instanceof Promise) {
                            parsedMarkdown.then(html => { aiMessageBubble.innerHTML = html; });
                        } else {
                            aiMessageBubble.innerHTML = parsedMarkdown;
                        }
                        chatWindow.scrollTop = chatWindow.scrollHeight;
                    }
                } catch (e) { /* Ignore partial JSON errors */ }
            }
        }
        return { fullResponse, finalDifyId };
    }

    // --- Corrected handleDocumentUpload and analyzeDocument Logic ---

    async function analyzeDocument(userId: string, filename: string): Promise<string> {
        try {
            const response = await fetch(`http://localhost:8000/analyze-document/${userId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to analyze document.');
            }

            const result = await response.json();
            return result.analysis;

        } catch (error) {
            console.error('Error during analysis:', error);
            throw error; // Re-throw the error to be caught by the caller
        }
    }

    async function handleDocumentUpload(event: Event) {
        const target = event.target as HTMLInputElement;
        const file = target.files?.[0];
        if (!file) return;

        const sendButton = document.getElementById('send-button') as HTMLButtonElement;
        const messageInput = document.getElementById('message-input') as HTMLInputElement;
        const originalButtonContent = sendButton.innerHTML;

        // --- Start of controlled UI state ---
        sendButton.innerHTML = `<div class="loader"></div>`;
        sendButton.disabled = true;
        messageInput.disabled = true;
        messageInput.placeholder = "Uploading document...";

        if (!appState.activeChatId) {
            await createNewChat();
        }
        const activeChat = getActiveChat();
        if (!activeChat) {
            // Reset UI on error
            sendButton.innerHTML = originalButtonContent;
            sendButton.disabled = false;
            messageInput.disabled = false;
            messageInput.placeholder = i18n.t('app_askAnything');
            return;
        }

        try {
            // Step 1: Upload the file
            const formData = new FormData();
            formData.append('file', file);
            const uploadResponse = await fetch(`http://localhost:8000/upload-document/${userIdentifier}`, {
                method: 'POST',
                body: formData,
            });

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                throw new Error(`Upload failed: ${errorData.detail || 'Unknown error'}`);
            }
            
            await uploadResponse.json(); // Consume the response body
            
            // Update chat state and persist
            activeChat.has_document = true;
            if (isGuestMode) {
                localStorage.setItem(GUEST_STORAGE_KEY, JSON.stringify(appState));
            } else {
                await supabase.from('chats').update({ has_document: true }).eq('id', activeChat.id);
            }

            // Step 2: Notify user and start analysis
            displayMessage(`📄 **${file.name}** uploaded. Now analyzing...`, 'ai');
            messageInput.placeholder = "Analyzing document...";
            renderSidebar();

            // Step 3: Call and wait for analysis to complete
            const analysisResult = await analyzeDocument(userIdentifier, file.name);

            // Step 4: Display the analysis result
            displayMessage(analysisResult, 'ai');

        } catch (error) {
            console.error("Document processing error:", error);
            const errorMessage = `Error: ${error instanceof Error ? error.message : 'An unknown error occurred. Please try again.'}`;
            displayMessage(errorMessage, 'ai');
        } finally {
            // --- End of controlled UI state: Reset UI ---
            target.value = ''; // Clear the file input
            sendButton.innerHTML = originalButtonContent;
            sendButton.disabled = false;
            messageInput.disabled = false;
            messageInput.placeholder = i18n.t('app_askAnything');
        }
    }

    function renderUserProfileLink() {
        if (!userProfileLink) return;
        if (isGuestMode) {
            userProfileLink.innerHTML = `<a href="/login" class="nav-button nav-button-primary" data-link>${i18n.t('app_signUpToSave')}</a>`;
        } else {
            const user = session?.user;
            const userInitial = user?.email?.charAt(0).toUpperCase() || 'P';
            userProfileLink.innerHTML = `<a href="/profile" data-link><div class="avatar user-avatar">${userInitial}</div><span>${user?.email}</span></a>`;
        }
    }

    async function initApp() {
        mermaid.initialize({ startOnLoad: false });
        function setupSpeechRecognition() {
            if (!recognition) {
                if (micButton) micButton.style.display = 'none';
                console.warn("Speech Recognition not supported in this browser.");
                return;
            }
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = i18n.getLanguage() === 'bn' ? 'bn-BD' : 'en-US';
            recognition.onstart = () => { isListening = true; micButton.classList.add('is-listening'); };
            recognition.onend = () => { isListening = false; micButton.classList.remove('is-listening'); };
            recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
                console.error("Speech recognition error", event.error);
                isListening = false;
                micButton.classList.remove('is-listening');
            };
            recognition.onresult = (event: SpeechRecognitionEvent) => {
                const transcript = event.results[0][0].transcript;
                messageInput.value = transcript;
                handleFormSubmit();
            };
            micButton.addEventListener('click', () => {
                if (synthesis.speaking) { synthesis.cancel(); }
                if (isListening) {
                    recognition.stop();
                } else {
                    recognition.start();
                }
            });
        }
        function toggleSidebar() {
            sidebar.classList.toggle('is-open');
            overlay.classList.toggle('is-open');
        }
        document.addEventListener('toggle-sidebar', toggleSidebar);
        overlay.addEventListener('click', toggleSidebar);
        conversationList.addEventListener('click', (e) => {
            if (window.innerWidth <= 900 && (e.target as HTMLElement).closest('.conversation-item')) {
                toggleSidebar();
            }
        });
        messageForm.addEventListener('submit', (e) => { e.preventDefault(); handleFormSubmit(); });
        chatWindow.addEventListener('click', (e) => {
            const target = e.target as HTMLElement;
            const queryItem = target.closest('.suggested-query-item');
            const followUpItem = target.closest('.follow-up-question');

            if (queryItem && queryItem.textContent) {
                messageInput.value = queryItem.textContent;
                handleFormSubmit();
            } else if (followUpItem && followUpItem.textContent) {
                messageInput.value = followUpItem.textContent;
                handleFormSubmit();
            }
        });
        newChatBtn.addEventListener('click', createNewChat);
        darkModeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            if (themeText) themeText.textContent = document.body.classList.contains('dark-mode') ? i18n.t('app_lightMode') : i18n.t('app_darkMode');
        });
        if (isGuestMode) {
            const guestNotice = document.createElement('div');
            guestNotice.style.cssText = 'background-color: var(--bg-soft); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 8px 12px; text-align: center; font-size: 14px; border-radius: 8px; margin-bottom: 16px;';
            guestNotice.innerHTML = `${i18n.t('app_guestNotice')} <a href="/login" data-link style="color: var(--accent-color-start); font-weight: 500;">${i18n.t('app_guestSignIn')}</a> ${i18n.t('app_guestToSave')}`;
            sidebar.prepend(guestNotice);
        }
        await loadState();
        if (appState.chats.length === 0) {
            await createNewChat();
        } else {
            await setActiveChat(appState.chats[0].id);
        }
        renderUserProfileLink();
        setupSpeechRecognition();
        sidebarLangSwitcher?.querySelector('.lang-en')?.addEventListener('click', () => {
            if (i18n.getLanguage() !== 'en') i18n.setLanguage('en');
        });
        sidebarLangSwitcher?.querySelector('.lang-bn')?.addEventListener('click', () => {
            if (i18n.getLanguage() !== 'bn') i18n.setLanguage('bn');
        });
        uploadDocBtn.addEventListener('click', () => { docFileInput.click(); });
        docFileInput.addEventListener('change', handleDocumentUpload);
    }
    await initApp();
}