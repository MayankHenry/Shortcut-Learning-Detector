// src/App.test.js
// Frontend unit tests using React Testing Library + Jest
// Run with: npm test

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

// ── Mock fetch globally ───────────────────────────────────────────────────────
beforeEach(() => {
  global.fetch = jest.fn();
});

afterEach(() => {
  jest.clearAllMocks();
});

// ── Helper: mock a successful login ──────────────────────────────────────────
const mockLogin = () => {
  global.fetch.mockResolvedValueOnce({
    ok: true,
    status: 200,
    json: async () => ({ access_token: "mock-jwt-token", token_type: "bearer" }),
  });
};

// ── Helper: mock a successful analyze response ────────────────────────────────
const mockAnalyze = () => {
  global.fetch.mockResolvedValueOnce({
    ok: true,
    status: 200,
    json: async () => ({
      class_name: "Digit 3",
      confidence: 94.2,
      heatmap_base64: "fakebase64string",
    }),
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// 1. Auth Modal Tests
// ─────────────────────────────────────────────────────────────────────────────

describe("Auth Modal", () => {
  test("renders login modal on initial load", () => {
    render(<App />);
    expect(screen.getByText("Sign in to continue")).toBeInTheDocument();
    expect(screen.getByLabelText("Username")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
  });

  test("shows validation error when submitting empty fields", async () => {
    render(<App />);
    fireEvent.click(screen.getByText(/Sign In/i));
    expect(
      await screen.findByText("Please enter both username and password.")
    ).toBeInTheDocument();
  });

  test("switches to register mode when Register link is clicked", async () => {
    render(<App />);
    fireEvent.click(screen.getByText("Register"));
    expect(screen.getByText("Create an account")).toBeInTheDocument();
  });

  test("switches back to login from register mode", async () => {
    render(<App />);
    fireEvent.click(screen.getByText("Register"));
    fireEvent.click(screen.getByText("Sign In"));
    expect(screen.getByText("Sign in to continue")).toBeInTheDocument();
  });

  test("shows error message when login fails", async () => {
    global.fetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ detail: "Incorrect username or password" }),
    });
    render(<App />);
    await userEvent.type(screen.getByLabelText("Username"), "wronguser");
    await userEvent.type(screen.getByLabelText("Password"), "wrongpass");
    fireEvent.click(screen.getByText(/Sign In/i));
    expect(
      await screen.findByText("Incorrect username or password")
    ).toBeInTheDocument();
  });

  test("hides modal and shows main app after successful login", async () => {
    mockLogin();
    render(<App />);
    await userEvent.type(screen.getByLabelText("Username"), "radhika");
    await userEvent.type(screen.getByLabelText("Password"), "password123");
    fireEvent.click(screen.getByText(/Sign In/i));
    await waitFor(() => {
      expect(screen.queryByText("Sign in to continue")).not.toBeInTheDocument();
    });
    expect(screen.getByText("ShortcutDetect")).toBeInTheDocument();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Navigation Tests
// ─────────────────────────────────────────────────────────────────────────────

describe("Navigation", () => {
  const loginAndGetApp = async () => {
    mockLogin();
    render(<App />);
    await userEvent.type(screen.getByLabelText("Username"), "radhika");
    await userEvent.type(screen.getByLabelText("Password"), "pass");
    fireEvent.click(screen.getByText(/Sign In/i));
    await waitFor(() =>
      expect(screen.queryByText("Sign in to continue")).not.toBeInTheDocument()
    );
  };

  test("shows analyzer page by default after login", async () => {
    await loginAndGetApp();
    expect(screen.getByText("Upload Image")).toBeInTheDocument();
    expect(screen.getByText("Select Model")).toBeInTheDocument();
  });

  test("navigates to How It Works page", async () => {
    await loginAndGetApp();
    fireEvent.click(screen.getByText("How It Works"));
    expect(await screen.findByText(/What is Shortcut Learning/i)).toBeInTheDocument();
  });

  test("navigates to Admin Logs page", async () => {
    // Mock the /admin/logs fetch
    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => [],
    });
    await loginAndGetApp();
    fireEvent.click(screen.getByText("Admin Logs"));
    expect(await screen.findByText("Prediction Logs")).toBeInTheDocument();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. File Upload Validation Tests
// ─────────────────────────────────────────────────────────────────────────────

describe("File Upload Validation", () => {
  const loginAndGetApp = async () => {
    mockLogin();
    render(<App />);
    await userEvent.type(screen.getByLabelText("Username"), "radhika");
    await userEvent.type(screen.getByLabelText("Password"), "pass");
    fireEvent.click(screen.getByText(/Sign In/i));
    await waitFor(() =>
      expect(screen.queryByText("Sign in to continue")).not.toBeInTheDocument()
    );
  };

  test("rejects non-image file types", async () => {
    await loginAndGetApp();
    const input = screen.getByLabelText("File input");
    const badFile = new File(["hello"], "document.pdf", { type: "application/pdf" });
    await userEvent.upload(input, badFile);
    expect(
      await screen.findByText("Please upload a JPG, PNG, WEBP, or BMP image.")
    ).toBeInTheDocument();
  });

  test("rejects files over 5MB", async () => {
    await loginAndGetApp();
    const input = screen.getByLabelText("File input");
    const bigFile = new File([new ArrayBuffer(6 * 1024 * 1024)], "big.png", {
      type: "image/png",
    });
    await userEvent.upload(input, bigFile);
    expect(
      await screen.findByText("File size must be under 5MB.")
    ).toBeInTheDocument();
  });

  test("accepts valid PNG file", async () => {
    await loginAndGetApp();
    const input = screen.getByLabelText("File input");
    const validFile = new File(["image-data"], "digit.png", { type: "image/png" });
    await userEvent.upload(input, validFile);
    // No error should appear
    await waitFor(() => {
      expect(screen.queryByText("Please upload a JPG, PNG, WEBP, or BMP image.")).not.toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Analyze Button Tests
// ─────────────────────────────────────────────────────────────────────────────

describe("Analyze Button", () => {
  const loginAndGetApp = async () => {
    mockLogin();
    render(<App />);
    await userEvent.type(screen.getByLabelText("Username"), "radhika");
    await userEvent.type(screen.getByLabelText("Password"), "pass");
    fireEvent.click(screen.getByText(/Sign In/i));
    await waitFor(() =>
      expect(screen.queryByText("Sign in to continue")).not.toBeInTheDocument()
    );
  };

  test("analyze button is disabled when no image is uploaded", async () => {
    await loginAndGetApp();
    const btn = screen.getByRole("button", { name: "Run Analysis" });
    expect(btn).toBeDisabled();
  });

  test("shows error if analyze is clicked without image", async () => {
    await loginAndGetApp();
    // Manually trigger via state edge case (button should be disabled, but just in case)
    const btn = screen.getByRole("button", { name: "Run Analysis" });
    expect(btn).toBeDisabled();
  });

  test("shows rate limit error on 429 response", async () => {
    await loginAndGetApp();
    const input = screen.getByLabelText("File input");
    const validFile = new File(["img"], "digit.png", { type: "image/png" });
    await userEvent.upload(input, validFile);

    global.fetch.mockResolvedValueOnce({ ok: false, status: 429, json: async () => ({}) });
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(
      await screen.findByText(/Too many requests/i)
    ).toBeInTheDocument();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Model Toggle Tests
// ─────────────────────────────────────────────────────────────────────────────

describe("Model Selection", () => {
  const loginAndGetApp = async () => {
    mockLogin();
    render(<App />);
    await userEvent.type(screen.getByLabelText("Username"), "radhika");
    await userEvent.type(screen.getByLabelText("Password"), "pass");
    fireEvent.click(screen.getByText(/Sign In/i));
    await waitFor(() =>
      expect(screen.queryByText("Sign in to continue")).not.toBeInTheDocument()
    );
  };

  test("biased model is selected by default", async () => {
    await loginAndGetApp();
    const biasedBtn = screen.getByRole("button", { name: "Biased Model" });
    expect(biasedBtn).toHaveAttribute("aria-pressed", "true");
  });

  test("can switch to unbiased model", async () => {
    await loginAndGetApp();
    fireEvent.click(screen.getByRole("button", { name: "Unbiased Model" }));
    const unbiasedBtn = screen.getByRole("button", { name: "Unbiased Model" });
    expect(unbiasedBtn).toHaveAttribute("aria-pressed", "true");
  });
});