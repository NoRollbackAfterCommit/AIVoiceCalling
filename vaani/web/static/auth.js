// Console-side authentication (see vaani/api/auth.py for the server-side gate).
//
// People sign in at /login and carry a session cookie. The browser attaches it to
// same-origin requests by itself — including WebSocket handshakes — so there is
// nothing for this file to add to a request.
//
// The console deliberately does NOT use the shared API token. That token is a
// machine credential: the deploy script, the health probe, the simulated carrier.
// A browser holding one bypassed the sign-in screen entirely — writes succeeded
// while /api/auth/me reported nobody signed in, so pages rendered blank lists
// over a session that did not exist, and knowledge could be added without anyone
// logging in. Any token found in storage is discarded on load rather than used.
//
// Plain browser JS, no build step, no dependencies.
(function () {
  'use strict';

  var LEGACY_KEY = 'vaani_api_token';
  var originalFetch = window.fetch;
  var redirecting = false;

  // Left over from before the console had accounts. Removing it on load is what
  // stops an existing browser quietly carrying on with admin rights.
  function discardLegacyToken() {
    try {
      window.localStorage.removeItem(LEGACY_KEY);
    } catch (e) {
      // localStorage can throw in a private window or with storage disabled.
      // Nothing sane to do, and nothing here depends on it succeeding.
    }
  }

  // Same-origin sockets carry the session cookie without help.
  function wsUrl(path) {
    var scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    return scheme + '://' + location.host + path;
  }

  // A rejected request means no usable session. Send them to sign in, once,
  // remembering where they were so they land back on it.
  function goToLogin() {
    if (redirecting) return;
    redirecting = true;
    var here = location.pathname + location.search;
    location.href = '/login?next=' + encodeURIComponent(here);
  }

  function me() {
    return originalFetch.call(window, '/api/auth/me', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }

  function signOut() {
    return originalFetch.call(window, '/api/auth/logout',
      { method: 'POST', credentials: 'same-origin' })
      .catch(function () { /* going to the login page regardless */ })
      .then(function () { location.href = '/login'; });
  }

  function isSameOriginApiPath(urlString) {
    var resolved;
    try {
      resolved = new URL(urlString, location.href);
    } catch (e) {
      return null;
    }
    if (resolved.origin !== location.origin) return null;
    if (resolved.pathname.indexOf('/api/') !== 0) return null;
    return resolved;
  }

  // Hide what the role cannot use, and give every page a way out. Driven by
  // data-roles attributes on the links themselves, so a page adds a link without
  // this file learning about it.
  function applyIdentity(user) {
    var header = document.querySelector('header');
    if (!header) return;

    Array.prototype.forEach.call(header.querySelectorAll('[data-roles]'), function (el) {
      var allowed = el.getAttribute('data-roles').split(/[,\s]+/);
      el.hidden = !(!user || allowed.indexOf(user.role) >= 0);
    });

    if (header.querySelector('[data-signout]')) return;
    var who = document.createElement('span');
    who.className = 'tag';
    who.setAttribute('data-signout', '');
    who.style.cssText = 'margin-left:14px;cursor:pointer';
    who.title = user ? 'Signed in as ' + user.email + ' — click to sign out' : 'Sign in';
    who.textContent = user ? (user.name || user.email) + ' · sign out' : 'Sign in';
    who.addEventListener('click', function () {
      if (user) { signOut(); } else { location.href = '/login'; }
    });
    header.appendChild(who);
  }

  function mountIdentity() {
    return me().then(function (user) {
      window.vaaniAuth.user = user;
      applyIdentity(user);
      if (user) return user;
      // Nobody is signed in. On an open deployment — the zero-config laptop
      // demo — that is normal and everything still works. On a guarded one it
      // is a dead end, and a page rendering nothing is worse than a login form,
      // so ask a guarded endpoint which of the two this is and let the 401
      // handling below decide.
      return window.fetch('/api/agents', { credentials: 'same-origin' })
        .then(function () { return null; })
        .catch(function () { return null; });
    });
  }

  window.vaaniAuth = {
    wsUrl: wsUrl,
    me: me,
    signOut: signOut,
    mountIdentity: mountIdentity,
    user: null,
  };

  window.fetch = function (input, init) {
    var urlString = input instanceof Request ? input.url : String(input);
    if (!isSameOriginApiPath(urlString)) {
      return originalFetch.call(window, input, init);
    }
    // Same-origin API call: the cookie rides along on its own. All this wrapper
    // does now is notice when there is no usable session.
    return originalFetch.call(window, input, init).then(function (response) {
      if (response.status === 401) goToLogin();
      return response;
    });
  };

  discardLegacyToken();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountIdentity);
  } else {
    mountIdentity();
  }
})();
