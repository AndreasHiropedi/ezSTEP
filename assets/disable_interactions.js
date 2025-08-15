function disablePageInteractions(modalType, modalIndex) {
    document.body.style.pointerEvents = 'none';

    var inputString = modalType;
    var element = findElementByIdString(inputString);

    if (element) {
        console.log("Element found");
        element.style.pointerEvents = 'all';
    } else {
        console.log("Element not found for formatted string:", objectToFormattedString(stringToObject(inputString)));
    }
}

function enablePageInteractions() {
    console.log('Enabling page interactions.');
    document.body.style.pointerEvents = 'all';
}

function objectToJsonString(obj) {
    return JSON.stringify(obj);
}

function stringToObject(str) {
    var parts = str.split('-');
    var index = parts.pop();
    var type = parts.join('-');
    return { index: parseInt(index, 10), type: type };
}

function findElementByIdString(str) {
    var obj = stringToObject(str);
    var jsonString = objectToJsonString(obj);
    return document.getElementById(jsonString);
}













