import React, { useState } from 'react';
import styled from 'styled-components';

// Banker blue theme colors
const bankerBg = "#354F64";
const bankerAccent = "#5884A7";
const bankerText = "#F9F9F9";
const bankerPanel = "#223142";

const PageContainer = styled.div`
  background-color: ${bankerBg};
  color: ${bankerText};
  width: 100%;
  min-height: 100vh;
  padding: 20px;
  overflow-y: auto;
`;

const SidePanel = styled.div`
  border: 1px solid ${bankerAccent};
  padding: 10px;
  border-radius: 8px;
  background-color: ${bankerPanel};
  color: ${bankerText};
  margin-bottom: 20px;
`;

const ToggleButton = styled.button`
  background-color: ${bankerAccent};
  color: ${bankerText};
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  margin-bottom: 10px;
  font-weight: bold;

  &:hover {
    background-color: ${bankerBg};
  }
`;

const CollapsibleContent = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
`;

const TextContent = styled.div`
  flex: 1;
  margin-right: 20px;
`;

const VideoWrapper = styled.div`
  flex-shrink: 0;
  width: 40%;
`;

const SearchForm = styled.form`
  margin: 32px 0 24px 0;
  display: flex;
  align-items: center;
  gap: 12px;
`;

const SearchLabel = styled.label`
  margin-right: 8px;
  font-weight: bold;
  color: ${bankerText};
`;

const SearchInput = styled.input`
  padding: 8px;
  border-radius: 4px;
  border: 1px solid ${bankerAccent};
  width: 350px;
  background: #406080;
  color: ${bankerText};
`;

const SearchButton = styled.button`
  background-color: ${bankerAccent};
  color: ${bankerText};
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  cursor: pointer;
  font-weight: bold;

  &:hover {
    background-color: ${bankerBg};
  }
`;

const ResultBox = styled.div`
  background: ${bankerPanel};
  color: ${bankerText};
  border: 1px solid ${bankerAccent};
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 24px;
  white-space: pre-wrap;
`;

const TwoColumnContainer = styled.div`
  display: flex;
  gap: 32px;
  width: 100%;
  @media (max-width: 900px) {
    flex-direction: column;
    gap: 0;
  }
`;

const LeftColumn = styled.div`
  flex: 2;
  min-width: 320px;
`;

const RightColumn = styled.div`
  flex: 1;
  min-width: 320px;
  background: ${bankerPanel};
  border: 1px solid ${bankerAccent};
  border-radius: 8px;
  padding: 20px;
  color: ${bankerText};
  font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace;
  font-size: 0.98rem;
  white-space: pre-wrap;
  overflow-x: auto;
`;

const CodeTitle = styled.div`
  font-weight: bold;
  color: ${bankerAccent};
  margin-bottom: 12px;
`;

const Investments = () => {
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [searchText, setSearchText] = useState("advise as to my financial situation");
  const [searchResult, setSearchResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSearchResult("");
    try {
      // 1. Fetch stock info for the customer
      const stockInfoResp = await fetch("https://oracleai-financial.org/financial/stockinfoforcustid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}) // Add customer id if needed, e.g. { customerId }
      });
      let stockInfo = "";
      if (stockInfoResp.ok) {
        stockInfo = await stockInfoResp.text();
      }

      // 2. Append the stock info to the prompt
      const prompt =
        searchText +
        " based on vanguard projections and the list of stocks purchases I am aslo sending (assume the stock name indicates the industry etc.). don't ask me to provide any other information\n\n" +
        stockInfo;

      // 3. Query endpoint
      const response = await fetch("https://oracleai-financial.org/financial/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: prompt })
      });
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const data = await response.json();
      setSearchResult(data.answer || "No answer found.");
    } catch (err) {
      setSearchResult("❌ Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const codeSnippet = `// Oracle Database Vector Search (PL/SQL)
SELECT *
FROM financial_docs
WHERE VECTOR_SEARCH('compliance', :query)
ORDER BY score DESC
FETCH FIRST 5 ROWS ONLY;

// Call MCP from PL/SQL
DECLARE
  result VARCHAR2(4000);
BEGIN
  result := MCP.GET_MARKET_DATA('AAPL');
  DBMS_OUTPUT.PUT_LINE(result);
END;
`;

  return (
    <PageContainer>
      <h2>Process: Get personal financial insights</h2>
      <h2>Tech: Vector Search, AI Agents and MCP</h2>
      <h2>Reference: DMCC</h2>

      {/* Collapsible SidePanel */}
      <SidePanel>
        <ToggleButton onClick={() => setIsCollapsed(!isCollapsed)}>
          {isCollapsed ? 'Show Developer Details' : 'Hide Developer Details'}
        </ToggleButton>
        {!isCollapsed && (
          <CollapsibleContent>
            <TextContent>
              <div>
                <a
                  href="https://paulparkinson.github.io/converged/microservices-with-converged-db/workshops/freetier-financial/index.html"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: bankerAccent, textDecoration: 'none' }}
                >
                  Click here for workshop lab and further information
                </a>
              </div>
              <div>
                <a
                  href="https://github.com/paulparkinson/oracle-ai-for-sustainable-dev/tree/main/financial"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: bankerAccent, textDecoration: 'none' }}
                >
                  Direct link to source code on GitHub
                </a>
              </div>
              <div>
                <a
                  href="http://141.148.204.74:8080"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: bankerAccent, textDecoration: 'none' }}
                >
                   AI Agents Backend
                </a>
              </div>
              <h4>Financial Process:</h4>
              <ul>
                <li>Generate financial insights using Oracle Database and AI Agents for private financial data, compliance docs, and market analysis</li>
              </ul>
              <h4>Developer Notes:</h4>
              <ul>
                <li>Uses Oracle Database for RAG with private financial</li>
                <li>Uses Oracle Database for vector searches of compliance pdfs</li>
                <li>Uses MCP for real-time market data</li>
              </ul>
              <h4>Differentiators:</h4>
              <ul>
                <li>Vector processing in the same database and with other business data (structured and unstructured)</li>
                <li>Call MCP from within the database using Java, JavaScript, or PL/SQL</li>
              </ul>
            </TextContent>
            <VideoWrapper>
              <h4>Walkthrough Video:</h4>
              <iframe
                width="100%"
                height="315"
                src="https://www.youtube.com/embed/fijnYAQ8zlk" 
                title="YouTube video player"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                style={{ borderRadius: '8px', border: `1px solid ${bankerAccent}` }}
              ></iframe>
            </VideoWrapper>
          </CollapsibleContent>
        )}
      </SidePanel>

      {/* Search and Code Snippet Section */}
      <TwoColumnContainer>
        <LeftColumn>
          <SearchForm onSubmit={handleSearch}>
            <SearchLabel htmlFor="searchText">Search:</SearchLabel>
            <SearchInput
              type="text"
              id="searchText"
              name="searchText"
              value={searchText}
              onChange={e => setSearchText(e.target.value)}
            />
            <SearchButton type="submit" disabled={loading}>
              {loading ? "Searching..." : "Search"}
            </SearchButton>
          </SearchForm>
          {searchResult && (
            <ResultBox>
              <strong>Result:</strong>
              <div>{searchResult}</div>
            </ResultBox>
          )}
        </LeftColumn>
        <RightColumn>
          <CodeTitle>Sample Vector Search & MCP Source Code</CodeTitle>
          <code>
            {codeSnippet}
          </code>
        </RightColumn>
      </TwoColumnContainer>
    </PageContainer>
  );
};

export default Investments;
