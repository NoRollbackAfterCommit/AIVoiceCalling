// Console-side authentication (see vaani/api/auth.py for the server-side gate).
//
// People sign in at /login and carry a session cookie, which the browser attaches to
// same-origin requests — including WebSocket handshakes — without any help from here.
// The stored API token remains supported for a machine or an operator pasting one in,
// and is tried first when present; a 401 with no session sends the person to sign in
// rather than asking them to paste a master key into a browser prompt.
// Plain browser JS, no build step, no dependencies — loaded directly by <script src="/static/auth.js">.
(function () {
  'use strict';

  var STORAGE_KEY = 'vaani_api_token';

  // localStorage can throw (private windows, storage disabled) — never let that break the console.
  function token() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      return null;
    }
  }

  function set(t) {
    try {
      window.localStorage.setItem(STORAGE_KEY, t);
    } catch (e) {
      // ignore — nothing sane to do if storage is unavailable
    }
  }

  function clear() {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      // ignore
    }
  }

  // Browsers can't send headers on a WebSocket handshake, so the token rides as a query param.
  function wsUrl(path) {
    var scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    var url = scheme + '://' + location.host + path;
    var t = token();
    if (t) {
      var sep = path.indexOf('?') >= 0 ? '&' : '?';
      url += sep + 'token=' + encodeURIComponent(t);
    }
    return url;
  }



  // --- fetch wrapper -------------------------------------------------------
  // Only same-origin /api/* requests get the Authorization header; everything else
  // (other origins, /static/*, /ws/* which fetch never touches) passes through untouched.
  var originalFetch = window.fetch;
  // A page fires several /api calls on load. When all of them get 401 at once,
  // the user must see one prompt, not one per request; they share this.
  var pendingPrompt = null;

  var redirecting = false;

  // A rejected request means no usable session. Send them to sign in, once,
  // remembering where they were so they land back on it.
  function goToLogin() {
    if (redirecting) return;
    redirecting = true;
    clear();
    var here = location.pathname + location.search;
    location.href = "/login?next=" + encodeURIComponent(here);
  }

  function me() {
    return originalFetch.call(window, "/api/auth/me", { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }

  function signOut() {
    return originalFetch.call(window, "/api/auth/logout",
      { method: "POST", credentials: "same-origin" })
      .catch(function () { /* going to the login page regardless */ })
      .then(function () { clear(); location.href = "/login"; });
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

  function requestUrlString(input) {
    if (input instanceof Request) return input.url;
    return String(input);
  }

  // Build a new Request with an Authorization header added, preserving everything else
  // (method, body, existing headers, credentials, etc.) from the original input/init pair.
  function withAuthHeader(input, init, t) {
    var headers;
    if (init && init.headers !== undefined) {
      headers = new Headers(init.headers);
    } else if (input instanceof Request) {
      headers = new Headers(input.headers);
    } else {
      headers = new Headers();
    }
    headers.set('Authorization', 'Bearer ' + t);

    if (input instanceof Request) {
      var merged = Object.assign({}, init || {});
      merged.headers = headers;
      return new Request(input, merged);
    }
    var newInit = Object.assign({}, init || {});
    newInit.headers = headers;
    return [input, newInit];
  }

  // Hide what the role cannot use, and give every page a way out.
  // Driven by data-role attributes on the links themselves, so a page adds a
  // link without this file learning about it.
  function applyIdentity(user) {
    var header = document.querySelector("header");
    if (!header) return;

    Array.prototype.forEach.call(header.querySelectorAll("[data-roles]"), function (el) {
      var allowed = el.getAttribute("data-roles").split(/[,\s]+/);
      var visible = !user || allowed.indexOf(user.role) >= 0;
      el.hidden = !visible;
    });

    if (header.querySelector("[data-signout]")) return;
    var who = document.createElement("span");
    who.className = "tag";
    who.setAttribute("data-signout", "");
    who.style.cssText = "margin-left:14px;cursor:pointer";
    who.title = user ? "Signed in as " + user.email + " — click to sign out" : "Sign in";
    who.textContent = user ? (user.name || user.email) + " · sign out" : "Sign in";
    who.addEventListener("click", function () {
      if (user) { signOut(); } else { location.href = "/login"; }
    });
    header.appendChild(who);
  }

  function mountIdentity() {
    me().then(function (user) {
      window.vaaniAuth.user = user;
      applyIdentity(user);
      if (user) return;
      // Nobody is signed in. That is normal on an open deployment and for a
      // machine holding the token, and it is a dead end for an operator — so
      // ask a guarded endpoint which of the three this is, through the wrapper,
      // and let its 401 handling send them to sign in. A page that renders
      // nothing is worse than a login form.
      window.fetch("/api/agents", { credentials: "same-origin" }).catch(function () {});
    });
  }

  window.vaaniAuth = {
    token: token, set: set, clear: clear, wsUrl: wsUrl,
    me: me, signOut: signOut, mountIdentity: mountIdentity, user: null,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountIdentity);
  } else {
    mountIdentity();
  }

  window.fetch = function (input, init) {
    var urlString = requestUrlString(input);
    var resolved = isSameOriginApiPath(urlString);

    if (!resolved) {
      return originalFetch.call(window, input, init);
    }

    // A Request's body can be read once; keep a copy for the retry before
    // the first attempt consumes it.
    var retryInput = input instanceof Request ? input.clone() : input;

    var t = token();
    var authedArgs = t ? withAuthHeader(input, init, t) : null;

    var doFetch = function () {
      if (authedArgs) {
        if (authedArgs instanceof Request) return originalFetch.call(window, authedArgs);
        return originalFetch.call(window, authedArgs[0], authedArgs[1]);
      }
      return originalFetch.call(window, input, init);
    };

    return doFetch().then(function (response) {
      if (response.status !== 401) return response;
      // Neither a session nor a usable token. An open server never 401s, so
      // this is the first moment signing in is warranted.
      goToLogin();
      return response;
    });
  };
})();
