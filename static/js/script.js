const claimInput =
    document.getElementById("claim");

const verifyButton =
    document.getElementById("verify-btn");

const buttonText =
    document.getElementById("button-text");

const loader =
    document.getElementById("loader");

const characterCount =
    document.getElementById("character-count");

const resultSection =
    document.getElementById("result-section");

const errorBox =
    document.getElementById("error-box");

const verdictElement =
    document.getElementById("verdict");

const confidenceElement =
    document.getElementById("confidence");

const claimResultElement =
    document.getElementById("claim-result");

const reasonElement =
    document.getElementById("reason");

const evidenceContainer =
    document.getElementById("evidence-container");


/* CHARACTER COUNT */

claimInput.addEventListener("input", () => {

    const length =
        claimInput.value.length;

    characterCount.textContent =
        `${length} characters`;

});


/* VERIFY BUTTON */

verifyButton.addEventListener(
    "click",
    verifyClaim
);


async function verifyClaim() {

    const claim =
        claimInput.value.trim();


    hideError();


    if (!claim) {

        showError(
            "Please enter a claim first."
        );

        claimInput.focus();

        return;
    }


    setLoading(true);

    resultSection.classList.add("hidden");


    try {

        const response =
            await fetch("/analyze", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    claim: claim
                })

            });


        const data =
            await response.json();


        if (!response.ok ||
            !data.success) {

            throw new Error(
                data.error ||
                "Verification failed."
            );

        }


        displayResult(
            data.result
        );


    } catch (error) {

        console.error(error);

        showError(
            error.message ||
            "Something went wrong."
        );

    } finally {

        setLoading(false);

    }

}


/* DISPLAY RESULT */

function displayResult(result) {

    resultSection.classList.remove(
        "hidden"
    );


    verdictElement.textContent =
        (result.verdict ||
        "UNCLEAR").toUpperCase();


    const confidence =
        Number(result.confidence || 0);


    confidenceElement.textContent =
        `${Math.round(
            confidence * 100
        )}%`;


    claimResultElement.textContent =
        result.claim || "";


    reasonElement.textContent =
        result.reason ||
        "No explanation provided.";


    displayEvidence(
        result.evidence || []
    );


    resultSection.scrollIntoView({
        behavior: "smooth"
    });

}


/* EVIDENCE */

function displayEvidence(evidence) {

    evidenceContainer.innerHTML = "";


    if (!evidence.length) {

        evidenceContainer.innerHTML = `
            <div class="evidence-card">
                <p class="evidence-text">
                    No evidence was retrieved.
                </p>
            </div>
        `;

        return;
    }


    evidence.forEach(item => {

        const card =
            document.createElement(
                "div"
            );


        card.className =
            "evidence-card";


        const similarity =
            item.similarity !== undefined
                ? `${item.similarity}%`
                : "N/A";


        card.innerHTML = `
            <div class="evidence-top">

                <span class="evidence-type">
                    ${escapeHTML(
                        item.type ||
                        "Evidence"
                    )}
                </span>

                <span class="similarity">
                    Similarity:
                    ${escapeHTML(
                        similarity
                    )}
                </span>

            </div>

            <p class="evidence-text">
                ${escapeHTML(
                    item.text || ""
                )}
            </p>

            <p class="source">
                Source:
                ${escapeHTML(
                    item.source ||
                    "Unknown"
                )}
            </p>
        `;


        evidenceContainer.appendChild(
            card
        );

    });

}


/* LOADING */

function setLoading(isLoading) {

    verifyButton.disabled =
        isLoading;


    if (isLoading) {

        buttonText.textContent =
            "Analyzing...";

        loader.classList.remove(
            "hidden"
        );

    } else {

        buttonText.textContent =
            "Verify Claim";

        loader.classList.add(
            "hidden"
        );

    }

}


/* ERROR */

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.classList.remove(
        "hidden"
    );

}


function hideError() {

    errorBox.classList.add(
        "hidden"
    );

}


/* SECURITY */

function escapeHTML(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");

}