// Console-side support for VAANI_API_TOKEN (see vaani/api/auth.py for the server-side gate).
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

  window.vaaniAuth = { token: token, set: set, clear: clear, wsUrl: wsUrl };

  // --- fetch wrapper -------------------------------------------------------
  // Only same-origin /api/* requests get the Authorization header; everything else
  // (other origins, /static/*, /ws/* which fetch never touches) passes through untouched.
  var originalFetch = window.fetch;
  // A page fires several /api calls on load. When all of them get 401 at once,
  // the user must see one prompt, not one per request; they share this.
  var pendingPrompt = null;

  function promptOnce() {
    if (!pendingPrompt) {
      pendingPrompt = Promise.resolve().then(function () {
        clear();
        var entered = window.prompt(
          'This Vaani server requires an API token. Paste it to continue.'
        );
        if (entered) set(entered);
        return entered;
      });
      pendingPrompt.then(function () { pendingPrompt = null; }, function () { pendingPrompt = null; });
    }
    return pendingPrompt;
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

      // The stored token (if any) was rejected. An open server never 401s, so
      // this is the first moment a prompt is warranted.
      return promptOnce().then(function (entered) {
        if (!entered) return response;
        var retryArgs = withAuthHeader(retryInput, init, entered);
        if (retryArgs instanceof Request) return originalFetch.call(window, retryArgs);
        return originalFetch.call(window, retryArgs[0], retryArgs[1]);
      });
    });
  };
})();
