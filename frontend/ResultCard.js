import React from "react";

const ResultCard = ({ result }) => {
  if (!result) return null;

  const { "Relevant Skills": skills, "Alignment with JD": alignment, Score, Verdict } = result;

  return (
    <div className="border rounded-xl p-4 shadow bg-white mt-4">
      <h3 className="text-lg font-semibold mb-2">Match Summary</h3>
      <div className="mb-2">
        <strong>Relevant Skills:</strong>
        <div className="flex flex-wrap gap-2 mt-1">
          {skills.map((skill, index) => (
            <span key={index} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
              {skill}
            </span>
          ))}
        </div>
      </div>
      <div className="mb-2">
        <strong>Alignment with JD:</strong> <p className="inline">{alignment}</p>
      </div>
      <div className="mb-2">
        <strong>Score:</strong> {Score} / 10
      </div>
      <div>
        <strong>Verdict:</strong>{" "}
        <span className={Verdict.startsWith("Yes") ? "text-green-700" : "text-red-700"}>
          {Verdict}
        </span>
      </div>
    </div>
  );
};

export default ResultCard;
