// A device's setup and pairing pages' backdrop (setup.ts): the chat's sea of characters filling the window, its butterfly
// above the page's card as it is above the message box in a new chat. When the page goes on to the chat (the device
// trusts this computer, or is let in) the butterfly dives into the water, as it does when a chat begins. Where no canvas
// can be drawn on, the logo stands where the butterfly would.
import logo from "./logo-thin.svg";
import { Ocean } from "./ocean";

const ocean = new Ocean(document.getElementById("sea")!, ".card", ""), up = document.createElement("i");
up.hidden = true;
document.body.append(up);
if (ocean.start()) ocean.butterfly(up);
else document.getElementById("stage")!.innerHTML = `<img class="logo" src="${logo}" alt="">`;
document.addEventListener("chrysalis-opening", () => up.remove());
