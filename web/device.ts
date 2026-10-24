import logo from "./logo.svg";
import { Ocean } from "./ocean";

const ocean = new Ocean(document.getElementById("sea")!, { box: ".card", hint: "", room: "#stage" }), up = document.createElement("i");
up.hidden = true;
document.body.append(up);
if (ocean.start()) ocean.butterfly(up);
else document.getElementById("stage")!.innerHTML = `<img class="logo" src="${logo}" alt="">`;
document.addEventListener("chrysalis-opening", () => up.remove());
