/* The page. Sends one question at a time and renders the answer as it streams.
 *
 * The access code lives in sessionStorage, not localStorage: closing the tab
 * forgets it, which is what you want on a machine someone else might use.
 */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var gate = $("gate"), agree = $("agree"), enter = $("enter"),
      codeRow = $("code-row"), code = $("code"), gateError = $("gate-error"),
      thread = $("thread"), composer = $("composer"), q = $("q"),
      send = $("send"), counter = $("counter"),
      dot = $("status-dot"), modelName = $("model-name");

  var accessCode = sessionStorage.getItem("accessCode") || "";
  var needsCode = false;
  var busy = false;

  /* ------------------------------------------------------------- gate --- */
  function showGate() {
    gate.hidden = false;
    codeRow.hidden = !needsCode;
    refreshEnter();
  }

  function refreshEnter() {
    enter.disabled = !agree.checked || (needsCode && !code.value.trim());
  }

  agree.addEventListener("change", refreshEnter);
  code.addEventListener("input", refreshEnter);

  enter.addEventListener("click", function () {
    if (needsCode) {
      accessCode = code.value.trim();
      sessionStorage.setItem("accessCode", accessCode);
    }
    localStorage.setItem("acceptedDisclaimer", "1");
    gate.hidden = true;
    q.focus();
  });

  /* ----------------------------------------------------------- health --- */
  fetch("/api/health")
    .then(function (r) { return r.json(); })
    .then(function (h) {
      needsCode = h.needs_code;
      dot.className = "dot " + (h.ok ? "up" : "down");
      dot.title = h.detail;
      modelName.textContent = h.model;
      if (!h.ok) addError(h.detail);
      if (!localStorage.getItem("acceptedDisclaimer") || (needsCode && !accessCode)) {
        showGate();
      }
    })
    .catch(function () {
      dot.className = "dot down";
      dot.title = "backend unreachable";
      addError("Can't reach the backend. Is the server still running?");
    });

  /* --------------------------------------------------------- composer --- */
  function autosize() {
    q.style.height = "auto";
    q.style.height = Math.min(q.scrollHeight, 160) + "px";
    counter.textContent = q.value.length + " / 500";
  }

  q.addEventListener("input", autosize);

  q.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composer.requestSubmit();
    }
  });

  Array.prototype.forEach.call(document.querySelectorAll(".eg"), function (b) {
    b.addEventListener("click", function () {
      q.value = b.textContent;
      autosize();
      composer.requestSubmit();
    });
  });

  composer.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = q.value.trim();
    if (!text || busy) return;
    q.value = "";
    autosize();
    ask(text);
  });

  /* ------------------------------------------------------------- view --- */
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function scroll() {
    var m = document.querySelector("main");
    m.scrollTop = m.scrollHeight;
  }

  function dropIntro() {
    var i = document.querySelector(".intro");
    if (i) i.remove();
  }

  function addError(msg) {
    dropIntro();
    var turn = el("div", "turn");
    turn.appendChild(el("div", "a err", msg));
    thread.appendChild(turn);
    scroll();
  }

  function setBusy(on) {
    busy = on;
    send.disabled = on;
    q.disabled = on;
  }

  /* -------------------------------------------------------------- ask --- */
  function ask(question) {
    dropIntro();
    setBusy(true);

    var turn = el("div", "turn");
    turn.appendChild(el("div", "q", question));

    var status = el("div", "waiting");
    status.appendChild(el("div", "spinner"));
    status.appendChild(el("span", null, "Thinking…"));
    turn.appendChild(status);

    thread.appendChild(turn);
    scroll();

    var answer = null;
    var started = Date.now();

    fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Access-Code": accessCode
      },
      body: JSON.stringify({ question: question })
    }).then(function (res) {
      if (!res.ok) {
        return res.json().then(function (j) {
          throw new Error(j.detail || ("Request failed (" + res.status + ")"));
        }, function () {
          throw new Error("Request failed (" + res.status + ")");
        });
      }
      return readStream(res.body);
    }).catch(function (err) {
      if (status.parentNode) status.remove();
      if (!answer) {
        answer = el("div", "a err");
        turn.appendChild(answer);
      }
      answer.className = "a err";
      answer.textContent = err.message;
      scroll();
    }).then(function () {
      setBusy(false);
      q.focus();
    });

    function readStream(body) {
      var reader = body.getReader();
      var decoder = new TextDecoder();
      var buf = "";

      return (function pump() {
        return reader.read().then(function (r) {
          if (r.done) {
            finish();
            return;
          }
          buf += decoder.decode(r.value, { stream: true });
          var lines = buf.split("\n");
          buf = lines.pop();
          lines.forEach(handle);
          return pump();
        });
      })();
    }

    function handle(line) {
      if (!line.trim()) return;
      var msg;
      try { msg = JSON.parse(line); } catch (e) { return; }

      if (msg.type === "queued") {
        status.lastChild.textContent =
          msg.position === 1
            ? "One question ahead of you…"
            : msg.position + " questions ahead of you…";

      } else if (msg.type === "start") {
        status.lastChild.textContent = "Thinking…";

      } else if (msg.type === "delta") {
        if (!answer) {
          status.remove();
          answer = el("div", "a caret");
          turn.appendChild(answer);
        }
        answer.appendChild(document.createTextNode(msg.text));
        scroll();

      } else if (msg.type === "error") {
        if (status.parentNode) status.remove();
        if (!answer) {
          answer = el("div", "a err");
          turn.appendChild(answer);
        } else {
          answer.classList.remove("caret");
          answer.appendChild(el("span", "meta", msg.message));
          answer = null;
          return;
        }
        answer.textContent = msg.message;
        scroll();

      } else if (msg.type === "done") {
        finish(msg);
      }
    }

    function finish(msg) {
      if (status.parentNode) status.remove();
      if (!answer) return;
      answer.classList.remove("caret");
      var secs = msg && msg.seconds ? msg.seconds : (Date.now() - started) / 1000;
      var bits = [secs.toFixed(0) + "s"];
      if (msg && msg.output_tokens) bits.push(msg.output_tokens + " tokens");
      bits.push("not medical advice");
      answer.appendChild(el("span", "meta", bits.join(" · ")));
      answer = null;
      scroll();
    }
  }

  autosize();
})();
