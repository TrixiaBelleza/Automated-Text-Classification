//find and modify tall tweets

$('#wmd-input').on("input", function(){
	var t = $(this).val();
	console.log(t)
});

var tagrecom_div = document.createElement("div");
tagrecom_div.id = "tagrecom_div"
tagrecom_div.class = "tagrecom_div"
document.getElementById("tag-editor").appendChild(tagrecom_div);

var tagrecom_btn = document.createElement('a');
tagrecom_btn.id = "tagrecom_btn"
tagrecom_btn.appendChild(document.createTextNode("Tag Suggestions"));
tagrecom_btn.href = '#';
tagrecom_btn.id = "tagrecom_btn";
document.getElementById('tagrecom_div').appendChild(tagrecom_btn)


// if(document.getElementById('tagrecom_btn').onclick()
document.getElementById('tagrecom_btn').onclick = function(){
    console.log("Clicked!");
}
// chrome.runtime.sendMessage({message: "listeners"}, function(response) {
// });

