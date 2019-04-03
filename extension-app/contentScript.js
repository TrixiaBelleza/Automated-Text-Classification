/************************************************
Create the Tag Recommendation DIV that contains
the "Tag Suggestions" button

************************************************/
var tagrecom_div = document.createElement('div');
tagrecom_div.id = "tagrecom_div";
document.getElementById("tag-editor").appendChild(tagrecom_div);

var tagrecom_btn = document.createElement('a');
tagrecom_btn.id = "tagrecom_btn";
tagrecom_btn.appendChild(document.createTextNode("Tag Suggestions"));
tagrecom_btn.style.fontSize = 'medium';
document.getElementById('tagrecom_div').appendChild(tagrecom_btn)

/************************************************
Create the Modal where Tag Suggestions will be 
displayed.
************************************************/

var modal_content = document.createElement('div');
modal_content.id = "modal_content";
document.getElementById('tagrecom_div').appendChild(modal_content);

var modal = document.querySelector('#modal_content');
modal.style.display = 'none';

var tag_suggestions_div = document.getElementById('tag-suggestions');
tag_suggestions_div.parentNode.removeChild(tag_suggestions_div);

/********************************************************************************************
Create a loader inside the modal while waiting for the tag suggestions to be displayed.
********************************************************************************************/
var loader = document.createElement('img');
loader.src = chrome.runtime.getURL("simple-loader.gif");
loader.style.width = '30px';
loader.style.height = '30px';
document.getElementById('tagrecom_div').appendChild(loader);
loader.style.display = 'none';

/*****************************************************
If a tag suggestion (let's say, "python") is clicked
*****************************************************/
function clicked(e) {
	// Once a suggested tag is clicked, it will disappear on the modal
	clicked_suggested = document.querySelector('#' + e.target.id);
	clicked_suggested.style.display = 'none';

	document.getElementById('tageditor-replacing-tagnames--input').placeholder = '';

	if(e.target.id != "others") {
		//Add tag to the Tags bar of Stack Overflow
		var tagspan = document.createElement('span');
		tagspan.classList.add('s-tag', 'rendered-element');
		tagspan.appendChild(document.createTextNode(e.target.id));
		tagspan.id = e.target.id + "_span";
		document.getElementsByClassName('tag-editor s-input')[0].firstChild.appendChild(tagspan);

		// Create a small "X" button beside the tag name
		var removeTag = document.createElement('a');
		removeTag.classList.add('js-delete-tag', 's-tag--dismiss');
		removeTag.setAttribute("title", "Remove tag");
		document.getElementById(e.target.id + "_span").appendChild(removeTag);
		var removeTagSVG = document.createElement('svg');
		removeTagSVG.style.pointerEvents = 'none';
		removeTagSVG.classList.add('svg-icon', 'iconClearSm');
		removeTagSVG.setAttribute("height", "12");
		removeTagSVG.setAttribute("width", "12"); 
		removeTagSVG.setAttribute("viewBox", "0 0 14 14");
		removeTag.appendChild(removeTagSVG);
		var pathSVG = document.createElement('path');
		pathSVG.setAttribute("d", "M12 3.41L10.59 2 7 5.59 3.41 2 2 3.41 5.59 7 2 10.59 3.41 12 7 8.41 10.59 12 12 10.59 8.41 7z");
		removeTagSVG.appendChild(pathSVG);
	}
	else {
		alert("'Others' represent other labels not included in the scope of the study.")
	}
}

/****************************
When the modal is open
*****************************/
function openModal() {
	modal.style.display = 'block';
	loader.style.display = 'block';

	/*
	Get "input question" from Stack overflow using $('#wmd-input').val()
	Then send it to python flask server.
	Then, get response from server. The response will contain the predicted tags.
	Append each predicted tags to Modal.
	*/
	const text = $('#title').val() + ' ' + $('#wmd-input').val()
	$.ajax({
	    url: "https://trixiabells.pythonanywhere.com/_get_text_input/",
	    data: text,
	    type: "POST",
	    contentType : "application/json",
	    success: function(resp){
    	  var tagButton;
    	  for(i=0; i<resp.predicted_tags.length; i++){
    	  	// each predicted tag will be displayed as a button in the Modal

    	  	tagButton = document.createElement('a');
    	  	tagButton.classList.add('s-tag', 'rendered-element');
    	  	tagButton.id = resp.predicted_tags[i];
    	  	tagButton.appendChild(document.createTextNode(resp.predicted_tags[i]));
    	  	document.getElementById('modal_content').appendChild(tagButton);
    	  	document.getElementById('modal_content').appendChild(document.createElement('br'));
    	  	
    	  	// When a predicted tag is clicked:
    	  	tagButton.onclick = function(e) {
    	  		clicked(e);
    	  	}
    	  }
		  loader.style.display = 'none';
	    },
	    error: function(e, s, t) {
	      console.log(e);
	      console.log(s);
	      console.log(t);
		}
	});
}

$(document).ready(function() {
	document.getElementById('tagrecom_btn').onclick = function(){
		//Delete all contents of the modal everytime the "Tag Suggestions" button is clicked
		//This is needed para hindi dumodoble yung nadidisplay na predicted tags
		while (modal_content.firstChild) {
    		modal_content.removeChild(modal_content.firstChild);
		}
		$('#wmd-input').each(function(){
			openModal();
		});	
	}
});
