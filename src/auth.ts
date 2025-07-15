// src/auth.ts
import { supabase } from './supabaseClient';
import { type Session } from '@supabase/supabase-js';

// A simple store to hold the session state
let currentSession: Session | null = null;

export const auth = {
  // A function to get the current session without an async call
  getSession: () => currentSession,
  
  // A function to set the session, called by our main router
  setSession: (session: Session | null) => {
    currentSession = session;
  },

  // A helper to get the user ID
  getUserId: () => currentSession?.user?.id ?? null,
};

// Immediately check for a session on load
supabase.auth.getSession().then(({ data: { session } }) => {
    auth.setSession(session);
});