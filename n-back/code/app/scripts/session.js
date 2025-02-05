
window.addEventListener("load", setup);

let timeBreak = 1000;
let timeInfo = 1000;
let nextN = -1;

function setup() {
  eel.set_session_parameters();
}

eel.expose(setNextN);
function setNextN(nextN) {

  if(nextN > -1) {

    $("#info").html(nextN + "-back");
    setTimeout(function() {
      $("#info").html("START");

      setTimeout(function() {
        window.location.href = "block.html?session=true&n=" + nextN;
      }, (timeInfo/5)*2);
    }, (timeInfo/5)*3);
  }
  else {
    eel.end_session();
    window.location.href = "index.html";
  }

}

eel.expose(setTimeParameters);
function setTimeParameters(tBreak, tInfo) {
  timeBreak = tBreak;
  timeInfo = tInfo;

  showText();
}

function showText() {
  urlParams = new URLSearchParams(window.location.search);
  hasBreak = urlParams.get("break");

  if (hasBreak=="true") {

    $("#info").html("BREAK");
    setTimeout(function() {
      eel.get_next_n();
    }, timeBreak);
  }
  else {
    eel.get_next_n();
  }
}
