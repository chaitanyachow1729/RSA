$(document).ready(function(){
    signup();
})

function signup(){
    console.log("fuction called")
    $("#encrypt").click(function(){
        $.get("/encrypt").done(function (data) {
            console.log(data);
            if(data['success']!=undefined){
                $(".encrypted_msg").removeClass("hidden");
                $("div.cryption-msg").text(data["success"]);
            }
            else{
                $("div.cryption-msg").text(data["error"]);
            }
            
          }).fail(function () {
            alert("Encryption Failed");
          });
    })
    $("#decrypt").click(function(){
        $.get("/decrypt").done(function (data) {
            console.log(data)
            if(data['success']!=undefined){
                $(".decrypted_msg").removeClass("hidden");
                $("div.cryption-msg").text(data["success"]);
            }
            else{
                $("div.cryption-msg").text(data["error"]);
            }
          }).fail(function () {
            alert("Decryption Failed");
          });
    })
}
