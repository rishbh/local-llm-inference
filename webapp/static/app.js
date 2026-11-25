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
      dot = $("status-dot"), modelName = $("model-name"),
      attach = $("attach"), file = $("file"), strip = $("attach-strip"),
      thumb = $("attach-thumb"), sizeLabel = $("attach-size"),
      removeBtn = $("attach-remove");

  var accessCode = sessionStorage.getItem("accessCode") || "";
  var needsCode = false;
  var busy = false;

  /* The image waiting to be sent with the next question, or null.
   * { dataUrl: "data:image/jpeg;base64,...", b64: "...", bytes: 12345 } */
  var pending = null;

  /* The vision encoder works at 896px, so anything larger is uploaded and
   * then thrown away. 1024 leaves a margin without wasting a phone's upload. */
  var MAX_PX = 1024;
  var maxMb = 4;              /* replaced by whatever /api/health reports */

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
      /* The button appears only if the backend confirmed this model can read
       * images, so a text-only model never offers an upload that would 400. */
      attach.hidden = !h.images;
      if (h.max_image_mb) maxMb = h.max_image_mb;
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

  /* --------------------------------------------------------- picture --- */
  /* Shrink in the browser rather than on the server, for three reasons: a
   * 12 MP phone photo takes forever to upload over a home tunnel, the model
   * throws away everything above 896px anyway, and re-encoding through a
   * canvas drops the EXIF block - which is where a phone writes the GPS
   * coordinates of where the picture was taken. */
  function shrink(f, cb) {
    var url = URL.createObjectURL(f);
    var img = new Image();

    img.onload = function () {
      URL.revokeObjectURL(url);
      var w = img.naturalWidth, h = img.naturalHeight;
      if (!w || !h) { cb(new Error("That file is not an image this page can read.")); return; }

      var scale = Math.min(1, MAX_PX / Math.max(w, h));
      var canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(w * scale));
      canvas.height = Math.max(1, Math.round(h * scale));
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);

      try {
        cb(null, canvas.toDataURL("image/jpeg", 0.85));
      } catch (err) {
        cb(new Error("That image could not be prepared for upload."));
      }
    };

    img.onerror = function () {
      URL.revokeObjectURL(url);
      cb(new Error("That file is not an image this page can read."));
    };

    img.src = url;
  }

  function showPending() {
    strip.hidden = !pending;
    attach.classList.toggle("on", !!pending);
    if (pending) {
      thumb.src = pending.dataUrl;
      sizeLabel.textContent = (pending.bytes / 1024).toFixed(0) + " KB, resized";
    }
  }

  function clearPending() {
    pending = null;
    file.value = "";              /* so picking the same file again still fires */
    thumb.removeAttribute("src");
    showPending();
  }

  attach.addEventListener("click", function () { if (!busy) file.click(); });

  removeBtn.addEventListener("click", clearPending);

  file.addEventListener("change", function () {
    var f = file.files && file.files[0];
    if (!f) return;

    /* Checked before reading it: the resize below has to decode the whole
     * thing into memory, and a 60 MB file would do that on the user's phone. */
    if (f.size > maxMb * 1024 * 1024 * 4) {
      addError("That image is far too large. Pick one under " + maxMb + " MB.");
      clearPending();
      return;
    }

    shrink(f, function (err, dataUrl) {
      if (err) { addError(err.message); clearPending(); return; }
      var b64 = dataUrl.split(",")[1] || "";
      var bytes = Math.round(b64.length * 3 / 4);
      if (bytes > maxMb * 1024 * 1024) {
        addError("That image is still over " + maxMb + " MB after resizing.");
        clearPending();
        return;
      }
      pending = { dataUrl: dataUrl, b64: b64, bytes: bytes };
      showPending();
      q.focus();
    });
  });

  composer.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = q.value.trim();
    /* An image on its own is a legitimate question; the backend supplies the
     * words. Nothing at all is not. */
    if ((!text && !pending) || busy) return;
    var image = pending;
    q.value = "";
    clearPending();
    autosize();
    ask(text, image);
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
    attach.disabled = on;
  }

  /* -------------------------------------------------------------- ask --- */
  function ask(question, image) {
    dropIntro();
    setBusy(true);

    var turn = el("div", "turn");
    var bubble = el("div", "q");
    if (image) {
      /* A copy in the thread, so the conversation still reads correctly once
       * the composer has been cleared. */
      var shown = document.createElement("img");
      shown.src = image.dataUrl;
      shown.alt = "the image you attached";
      bubble.appendChild(shown);
    }
    bubble.appendChild(document.createTextNode(
      question || "(no question - just this image)"));
    turn.appendChild(bubble);

    var status = el("div", "waiting");
    status.appendChild(el("div", "spinner"));
    /* Reading a picture on a CPU takes a good while before any text appears,
     * so say so rather than letting it look stuck. */
    status.appendChild(el("span", null,
      image ? "Reading the image… this takes longer" : "Thinking…"));
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
      body: JSON.stringify(image
        ? { question: question, image: image.b64 }
        : { question: question })
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
        status.lastChild.textContent =
          image ? "Reading the image… this takes longer" : "Thinking…";

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
