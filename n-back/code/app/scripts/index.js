
window.addEventListener("load", setup);

function setup() {
  $("#start_session").click(function() {
    eel.start_session();
    window.location.href = "info.html?break=false";
  });

  $("#start_block").click(function() {
    window.location.href = "select.html";
  });

  $("#start_button").click(function() {
    $(this).hide("fast", function() {
      window.location.href = "block.html?session=false&n=" + $("#enter-n input").val();
    });
    $("#enter-n").hide();
  });
}
