
let n = 0
let isSession = false;

let timeSymbol = 1000;
let timePause = 0;
let targetKeyCode = 74;
let noTargetKeyCode = 70;

let charIdx = -1;


window.addEventListener("load", setup);

function setup() {
  getURLParams();
  eel.set_block_parameters();
  setupTargetKeys();
  startNBack();
}

function getURLParams() {
  urlParams = new URLSearchParams(window.location.search);
  n = urlParams.get("n");
  isSession = urlParams.get("session");
}

function loadBlock(block_n) {
  window.location.href = "block.html?n=" + block_n;
}

eel.expose(setTimeParameters);
function setTimeParameters(t0, t1) {
  timeSymbol = t0;
  timePause = t1;
}

function setupTargetKeys() {
  $("body").keydown(function(event) {
    if(event.which == targetKeyCode) {
      eel.target_pressed($("#cur_char").data("index"));
    } else if (event.which == noTargetKeyCode) {
      eel.no_target_pressed($("#cur_char").data("index"));
    }
  });
}

function startNBack() {
  $("#cur_char").html("");
  $("#cur_char").css("visibility", "visible");
  eel.start_n_back(n);
}

eel.expose(showSequence);
function showSequence(seq) {
  charIdx = -1;
  showNext(seq);
}

function showNext(seq) {

  if (seq.length > 0) {
    // LSL next character
    charIdx += 1;
    showCharacter(seq[0], charIdx);
    seq.shift();

    setTimeout(function() {
      showCharacter("");

      setTimeout(function() {
        showNext(seq);
      }, timePause);

    }, timeSymbol);
  }
  else {
    showCharacter("");
    eel.end_n_back();

    if (isSession=="true") {
      window.location.href = "info.html?break=true";
    } else {
      window.location.href = "index.html";
    }
  }
}

function showCharacter(c) {
  $("#cur_char").html(c);
  $("#cur_char").data("index", charIdx);
}

eel.expose(correctInputFeedback)
function correctInputFeedback() {
  $("body").addClass("correct");
  setTimeout(function() {
    $("body").removeClass("correct");
  }, 500);
}

eel.expose(wrongInputFeedback)
function wrongInputFeedback() {
  $("body").addClass("wrong");
  setTimeout(function() {
    $("body").removeClass("wrong");
  }, 500);
}
