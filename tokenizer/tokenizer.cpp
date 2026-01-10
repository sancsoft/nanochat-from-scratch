// test function
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <stdexcept>
#include <regex>
#include <filesystem>
#include <fstream>

#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <unicode/utypes.h>

static const char* SPLIT_PATTERN =
    "(?i:[sdmt]|ll|ve|re)|"
    "[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|"
    "\\p{N}{1,2}|"
    " ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|"
    "\\s*[\\r\\n]|"
    "\\s+(?!\\S)|"
    "\\s+";

inline std::uint64_t pack_pair(std::uint32_t a, std::uint32_t b) noexcept {
    return (std::uint64_t(a) << 32) | std::uint64_t(b);
}
inline std::uint32_t unpack_a(std::uint64_t k) noexcept { return std::uint32_t(k >> 32); }
inline std::uint32_t unpack_b(std::uint64_t k) noexcept { return std::uint32_t(k & 0xFFFFFFFFull); }

std::vector<std::uint32_t> to_byte_ids(const std::string& s) {
    std::vector<std::uint32_t> ids;
    ids.reserve(s.size());
    for (unsigned char ch : s) ids.push_back(std::uint32_t(ch)); // 0..255
    return ids;
}

std::unordered_map<std::uint64_t, std::int64_t>
count_adjacent_pairs(const std::vector<std::uint32_t>& ids) {
    std::unordered_map<std::uint64_t, std::int64_t> counts;
    if (ids.size() < 2) return counts;

    counts.max_load_factor(0.8f);
    counts.reserve(ids.size()); // rough

    for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
        counts[pack_pair(ids[i], ids[i + 1])] += 1;
    }
    return counts;
}

std::pair<std::uint64_t, std::int64_t>
most_frequent_pair(const std::unordered_map<std::uint64_t, std::int64_t>& counts) {
    std::uint64_t best_key = 0;
    std::int64_t best_count = 0;
    for (const auto& kv : counts) {
        if (kv.second > best_count) {
            best_count = kv.second;
            best_key = kv.first;
        }
    }
    return {best_key, best_count};
}

// Merge all non-overlapping occurrences of (a,b) into new_id.
void apply_merge(std::vector<std::uint32_t>& ids,
                        std::uint32_t a, std::uint32_t b, std::uint32_t new_id) {
    if (ids.size() < 2) return;

    std::vector<std::uint32_t> out;
    out.reserve(ids.size());

    std::size_t i = 0;
    while (i < ids.size()) {
        if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) {
            out.push_back(new_id);
            i += 2;
        } else {
            out.push_back(ids[i]);
            i += 1;
        }
    }
    ids.swap(out);
}

// Escape raw bytes for readable printing.
// Keeps normal ASCII printable. Escapes \n, \t, \r, \\, and quotes.
// Non-printable bytes become \xHH.
std::string escape_bytes(const std::string& s) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size());

    for (unsigned char c : s) {
        switch (c) {
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            case '\\': out += "\\\\"; break;
            case '\"': out += "\\\""; break;
            default:
                if (c >= 32 && c <= 126) {
                    out.push_back(static_cast<char>(c));
                } else {
                    out += "\\x";
                    out.push_back(hex[(c >> 4) & 0xF]);
                    out.push_back(hex[c & 0xF]);
                }
        }
    }
    return out;
}

// Print tokens as: "<raw>"(id)
void print_tokens_with_raw(const std::vector<std::uint32_t>& ids,
                                  const std::vector<std::string>& token_bytes) {
    for (std::uint32_t t : ids) {
        if (t < token_bytes.size()) {
            std::cout << "\"" << escape_bytes(token_bytes[t]) << "\"(" << t << ") ";
        } else {
            // Should not happen if token_bytes is maintained correctly
            std::cout << "\"?\"(" << t << ") ";
        }
    }
    std::cout << "\n";
}

std::vector<std::string> icu_tokenize_utf8(const std::string& input_utf8) {
    UErrorCode status = U_ZERO_ERROR;

    // UTF-8 -> ICU UnicodeString
    icu::UnicodeString input = icu::UnicodeString::fromUTF8(input_utf8);

    // Compile pattern
    icu::UnicodeString pat = icu::UnicodeString::fromUTF8(SPLIT_PATTERN);
    std::unique_ptr<icu::RegexPattern> pattern(icu::RegexPattern::compile(pat, 0, status));
    if (U_FAILURE(status) || !pattern) {
        throw std::runtime_error("ICU: failed to compile regex");
    }

    // Matcher
    std::unique_ptr<icu::RegexMatcher> m(pattern->matcher(input, status));
    if (U_FAILURE(status) || !m) {
        throw std::runtime_error("ICU: failed to create matcher");
    }

    std::vector<std::string> out;
    while (m->find(status)) {
        if (U_FAILURE(status)) break;

        int32_t start = m->start(status);
        int32_t end   = m->end(status);
        if (U_FAILURE(status)) break;

        icu::UnicodeString piece = input.tempSubStringBetween(start, end);
        std::string token;
        piece.toUTF8String(token);
        out.push_back(std::move(token));
    }

    if (U_FAILURE(status)) {
        throw std::runtime_error("ICU: regex find() failed");
    }

    return out;
}

std::vector<std::string> list_file_paths(const std::string& directory_path)
{
    std::vector<std::string> paths;

    if (!std::filesystem::exists(directory_path)) {
        throw std::runtime_error("Directory does not exist");
    }

    if (!std::filesystem::is_directory(directory_path)) {
        throw std::runtime_error("Path is not a directory");
    }

    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            paths.push_back(entry.path().string());
        }
    }

    return paths;
}

std::string read_file_utf8(const std::string& file_path)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string buffer;
    buffer.resize(static_cast<size_t>(size));

    if (size > 0) {
        file.read(&buffer[0], size);
    }

    return buffer;
}

extern "C" int train_tokenizer(const char* directory, const char* output_path, int vocab)
{
    if (!directory || !output_path) {
        std::cerr << "Null path received\n";
        return -1;
    }

    std::cout << "Directory: " << directory << std::endl;
    std::cout << "Output path: " << output_path << std::endl;
    std::cout << "Vocab size: " << vocab << std::endl;

    std::vector<std::string> file_paths = list_file_paths(directory);
    for(int i = 0; i< file_paths.size(); ++i)
    {
        std::cout << read_file_utf8(file_paths[i]) << std::endl;
    }

    // Token bytes table: id -> raw byte string for that token
    std::vector<std::string> token_bytes;
    token_bytes.reserve(4096);

    // Init 0..255 as single-byte tokens
    token_bytes.resize(256);
    for (std::uint32_t i = 0; i < 256; ++i) {
        token_bytes[i].push_back(static_cast<char>(static_cast<unsigned char>(i)));
    }

    // Step 1: bytes
    std::string text = "bababnamanmaanaannasndasd asnadn dsa ";
    std::vector<std::uint32_t> ids = to_byte_ids(text);

    std::cout << "Initial tokens:\n";
    print_tokens_with_raw(ids, token_bytes);

    std::uint32_t next_id = 256;

    // Do a few merges for demo
    for (int iter = 0; iter < 10; ++iter) {
        // Step 2: count pairs
        auto counts = count_adjacent_pairs(ids);
        if (counts.empty()) break;

        // Step 3: pick best
        auto [best_key, best_count] = most_frequent_pair(counts);
        if (best_count <= 1) {
            std::cout << "Stop: no pair occurs more than once.\n";
            break;
        }

        std::uint32_t left = unpack_a(best_key);
        std::uint32_t right = unpack_b(best_key);

        // Step 4: assign new token id
        std::uint32_t merged_id = next_id++;

        // Build raw bytes for merged token: bytes(left) + bytes(right)
        if (token_bytes.size() <= merged_id) token_bytes.resize(merged_id + 1);
        token_bytes[merged_id] = token_bytes[left] + token_bytes[right];

        // Report merge using raw strings too
        std::cout << "Merge " << iter << ": "
                  << "\"" << escape_bytes(token_bytes[left]) << "\"(" << left << ") + "
                  << "\"" << escape_bytes(token_bytes[right]) << "\"(" << right << ") -> "
                  << "\"" << escape_bytes(token_bytes[merged_id]) << "\"(" << merged_id << ")"
                  << "  freq=" << best_count << "\n";

        // Step 5: apply merge
        apply_merge(ids, left, right, merged_id);

        // Show updated sequence
        print_tokens_with_raw(ids, token_bytes);
    }

    return 0;
}
