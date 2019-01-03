//find and modify tall tweets

$('#wmd-input').on("input", function(){
	var t = $(this).val();
	console.log(t)
});

var tagrecom_div = document.createElement("div");
tagrecom_div.id = "tagrecom_div"
tagrecom_div.class = "tagrecom_div"
document.getElementById("tag-editor").appendChild(tagrecom_div);

var tagrecom_btn = document.createElement('button');
tagrecom_btn.id = "tagrecom_btn";
var tagrecom_btn_txt = document.createTextNode("Tag Suggestions");  
tagrecom_btn.appendChild(tagrecom_btn_txt); 
document.getElementById('tagrecom_div').appendChild(tagrecom_btn)


