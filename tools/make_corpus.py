"""Write the 512-token evaluation corpus used by the Stage 0 traces.

48 samples, 8 each across six domains (code, math, prose, factual, structured,
dialogue). The domain split is the whole point: question 4 measures how far
expert usage diverges *between* domains against a same-domain floor, and
question 5's policy ranking turns on how many distinct documents the cache
simulator sees. A corpus of one domain makes frequency policies look far better
than they are.

Every sample tokenizes to at least 512 tokens under the OLMoE tokenizer
(measured min 529, max 1704), so a 512-token trace window is fully populated
and no sequence is silently short. The text is model-generated rather than
scraped, which keeps the corpus redistributable and free of any risk that it
overlaps the model's training data in a way that would flatter the routing
statistics.

Usage:
    python tools/make_corpus.py [OUTPUT.jsonl]
"""

SAMPLES = [
    {"domain": "code", "text": """
import asyncio
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TokenType(Enum):
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    OPERATOR = "OPERATOR"
    KEYWORD = "KEYWORD"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    SEMICOLON = "SEMICOLON"
    COMMA = "COMMA"
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

@dataclass
class Position:
    line: int
    column: int
    offset: int

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = Position(1, 1, 0)
        self.current = self.source[0] if source else None
        self.tokens: List[Token] = []

    def peek(self, offset: int = 1) -> Optional[str]:
        idx = self.pos.offset + offset
        return self.source[idx] if idx < len(self.source) else None

    def advance(self) -> None:
        if self.current == '\\n':
            self.pos.line += 1
            self.pos.column = 1
        else:
            self.pos.column += 1
        self.pos.offset += 1
        self.current = self.source[self.pos.offset] if self.pos.offset < len(self.source) else None

    def skip_whitespace(self) -> None:
        while self.current and self.current.isspace():
            self.advance()

    def skip_comment(self) -> None:
        if self.current == '/' and self.peek(1) == '/':
            while self.current and self.current != '\\n':
                self.advance()
            self.advance()
        elif self.current == '/' and self.peek(1) == '*':
            self.advance()
            self.advance()
            while self.current:
                if self.current == '*' and self.peek(1) == '/':
                    self.advance()
                    self.advance()
                    break
                self.advance()

    def read_string(self, quote: str) -> str:
        value = ""
        self.advance()
        while self.current and self.current != quote:
            if self.current == '\\\\':
                self.advance()
                if self.current:
                    escape_map = {'n': '\\n', 't': '\\t', 'r': '\\r', '\\\\': '\\\\', quote: quote}
                    value += escape_map.get(self.current, self.current)
                    self.advance()
            else:
                value += self.current
                self.advance()
        if self.current == quote:
            self.advance()
        return value

    def read_number(self) -> str:
        value = ""
        while self.current and (self.current.isdigit() or self.current == '.'):
            value += self.current
            self.advance()
        return value

    def read_identifier(self) -> str:
        value = ""
        while self.current and (self.current.isalnum() or self.current == '_'):
            value += self.current
            self.advance()
        return value

    def tokenize(self) -> List[Token]:
        keywords = {"if", "else", "while", "for", "return", "function", "const", "let", "var", "class"}

        while self.current:
            self.skip_whitespace()
            if not self.current:
                break

            if self.current == '/':
                if self.peek(1) in ('/', '*'):
                    self.skip_comment()
                    continue
                else:
                    self.tokens.append(Token(TokenType.OPERATOR, '/', self.pos.line, self.pos.column))
                    self.advance()
            elif self.current in ('"', "'"):
                quote = self.current
                value = self.read_string(quote)
                self.tokens.append(Token(TokenType.STRING, value, self.pos.line, self.pos.column))
            elif self.current.isdigit():
                value = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, value, self.pos.line, self.pos.column))
            elif self.current.isalpha() or self.current == '_':
                value = self.read_identifier()
                token_type = TokenType.KEYWORD if value in keywords else TokenType.IDENTIFIER
                self.tokens.append(Token(token_type, value, self.pos.line, self.pos.column))
            elif self.current == '(':
                self.tokens.append(Token(TokenType.LPAREN, '(', self.pos.line, self.pos.column))
                self.advance()
            elif self.current == ')':
                self.tokens.append(Token(TokenType.RPAREN, ')', self.pos.line, self.pos.column))
                self.advance()
            elif self.current == '{':
                self.tokens.append(Token(TokenType.LBRACE, '{', self.pos.line, self.pos.column))
                self.advance()
            elif self.current == '}':
                self.tokens.append(Token(TokenType.RBRACE, '}', self.pos.line, self.pos.column))
                self.advance()
            elif self.current == ';':
                self.tokens.append(Token(TokenType.SEMICOLON, ';', self.pos.line, self.pos.column))
                self.advance()
            elif self.current == ',':
                self.tokens.append(Token(TokenType.COMMA, ',', self.pos.line, self.pos.column))
                self.advance()
            elif self.current in '+-*=<>!&|':
                op = self.current
                self.advance()
                if self.current in '=<>':
                    op += self.current
                    self.advance()
                self.tokens.append(Token(TokenType.OPERATOR, op, self.pos.line, self.pos.column - len(op)))
            else:
                self.advance()

        self.tokens.append(Token(TokenType.EOF, '', self.pos.line, self.pos.column))
        return self.tokens

async def tokenize_async(source: str) -> List[Token]:
    lexer = Lexer(source)
    return await asyncio.to_thread(lexer.tokenize)
"""},
    {"domain": "code", "text": """
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    timeout: Duration,
    buffer_size: usize,
    max_retries: u32,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        ConnectionConfig {
            timeout: Duration::from_secs(30),
            buffer_size: 8192,
            max_retries: 3,
        }
    }
}

pub struct Connection {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    config: ConnectionConfig,
    last_activity: Instant,
}

impl Connection {
    pub fn new(addr: &str, config: ConnectionConfig) -> io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(config.timeout))?;
        stream.set_write_timeout(Some(config.timeout))?;

        let reader = BufReader::new(stream.try_clone()?);
        let writer = BufWriter::new(stream.try_clone()?);

        Ok(Connection {
            stream,
            reader,
            writer,
            config,
            last_activity: Instant::now(),
        })
    }

    pub fn send(&mut self, data: &[u8]) -> io::Result<()> {
        self.writer.write_all(data)?;
        self.writer.flush()?;
        self.last_activity = Instant::now();
        Ok(())
    }

    pub fn recv(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let n = self.reader.read(buffer)?;
        self.last_activity = Instant::now();
        Ok(n)
    }

    pub fn is_idle(&self) -> bool {
        self.last_activity.elapsed() > self.config.timeout
    }
}

pub struct ConnectionPool {
    connections: Arc<Mutex<HashMap<String, VecDeque<Connection>>>>,
    config: ConnectionConfig,
}

impl ConnectionPool {
    pub fn new(config: ConnectionConfig) -> Self {
        ConnectionPool {
            connections: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    pub fn acquire(&self, addr: &str) -> io::Result<Connection> {
        let mut conns = self.connections.lock().unwrap();

        if let Some(queue) = conns.get_mut(addr) {
            while let Some(conn) = queue.pop_front() {
                if !conn.is_idle() {
                    return Ok(conn);
                }
            }
        }

        Connection::new(addr, self.config.clone())
    }

    pub fn release(&self, addr: String, conn: Connection) {
        let mut conns = self.connections.lock().unwrap();
        conns.entry(addr).or_insert_with(VecDeque::new).push_back(conn);
    }

    pub fn cleanup_idle(&self) {
        let mut conns = self.connections.lock().unwrap();
        for queue in conns.values_mut() {
            queue.retain(|conn| !conn.is_idle());
        }
    }
}

pub struct Server {
    listener: TcpListener,
    pool: Arc<ConnectionPool>,
}

impl Server {
    pub fn new(addr: &str) -> io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        Ok(Server {
            listener,
            pool: Arc::new(ConnectionPool::new(ConnectionConfig::default())),
        })
    }

    pub fn run(&self) -> io::Result<()> {
        for stream in self.listener.incoming() {
            let stream = stream?;
            let pool = Arc::clone(&self.pool);
            thread::spawn(move || {
                if let Err(e) = handle_client(stream, pool) {
                    eprintln!("Client error: {}", e);
                }
            });
        }
        Ok(())
    }
}

fn handle_client(stream: TcpStream, pool: Arc<ConnectionPool>) -> io::Result<()> {
    let mut buffer = [0u8; 1024];
    let mut reader = BufReader::new(&stream);

    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        println!("Received: {:?}", &buffer[..n]);
    }
    Ok(())
}
"""},
    {"domain": "code", "text": """
package main

import (
    "bufio"
    "context"
    "encoding/json"
    "flag"
    "fmt"
    "io"
    "log"
    "net/http"
    "os"
    "os/signal"
    "strings"
    "sync"
    "syscall"
    "time"
)

type HTTPClient struct {
    client    *http.Client
    baseURL   string
    headers   map[string]string
    mu        sync.RWMutex
    timeout   time.Duration
}

func NewHTTPClient(baseURL string, timeout time.Duration) *HTTPClient {
    return &HTTPClient{
        client: &http.Client{
            Timeout: timeout,
        },
        baseURL: strings.TrimSuffix(baseURL, "/"),
        headers: make(map[string]string),
        timeout: timeout,
    }
}

func (hc *HTTPClient) SetHeader(key, value string) {
    hc.mu.Lock()
    defer hc.mu.Unlock()
    hc.headers[key] = value
}

func (hc *HTTPClient) Get(ctx context.Context, path string) ([]byte, error) {
    req, err := http.NewRequestWithContext(ctx, http.MethodGet, hc.baseURL+path, nil)
    if err != nil {
        return nil, fmt.Errorf("create request: %w", err)
    }

    hc.mu.RLock()
    for k, v := range hc.headers {
        req.Header.Set(k, v)
    }
    hc.mu.RUnlock()

    resp, err := hc.client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("do request: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode >= 400 {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("status %d: %s", resp.StatusCode, body)
    }

    return io.ReadAll(resp.Body)
}

func (hc *HTTPClient) Post(ctx context.Context, path string, body interface{}) ([]byte, error) {
    var reader io.Reader
    switch v := body.(type) {
    case string:
        reader = strings.NewReader(v)
    case []byte:
        reader = strings.NewReader(string(v))
    default:
        data, err := json.Marshal(v)
        if err != nil {
            return nil, fmt.Errorf("marshal body: %w", err)
        }
        reader = strings.NewReader(string(data))
    }

    req, err := http.NewRequestWithContext(ctx, http.MethodPost, hc.baseURL+path, reader)
    if err != nil {
        return nil, fmt.Errorf("create request: %w", err)
    }

    req.Header.Set("Content-Type", "application/json")
    hc.mu.RLock()
    for k, v := range hc.headers {
        req.Header.Set(k, v)
    }
    hc.mu.RUnlock()

    resp, err := hc.client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("do request: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode >= 400 {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("status %d: %s", resp.StatusCode, body)
    }

    return io.ReadAll(resp.Body)
}

type CLI struct {
    client *HTTPClient
    reader *bufio.Reader
}

func NewCLI(client *HTTPClient) *CLI {
    return &CLI{
        client: client,
        reader: bufio.NewReader(os.Stdin),
    }
}

func (c *CLI) Run(ctx context.Context) error {
    fmt.Println("HTTP Client CLI. Type 'help' for commands.")

    for {
        fmt.Print("> ")
        line, err := c.reader.ReadString('\\n')
        if err != nil && err != io.EOF {
            return err
        }

        line = strings.TrimSpace(line)
        if line == "" {
            continue
        }

        parts := strings.Fields(line)
        cmd := parts[0]

        switch cmd {
        case "get":
            if len(parts) < 2 {
                fmt.Println("usage: get <path>")
                continue
            }
            body, err := c.client.Get(ctx, parts[1])
            if err != nil {
                fmt.Printf("error: %v\\n", err)
            } else {
                fmt.Println(string(body))
            }
        case "post":
            if len(parts) < 3 {
                fmt.Println("usage: post <path> <data>")
                continue
            }
            body, err := c.client.Post(ctx, parts[1], strings.Join(parts[2:], " "))
            if err != nil {
                fmt.Printf("error: %v\\n", err)
            } else {
                fmt.Println(string(body))
            }
        case "quit", "exit":
            return nil
        default:
            fmt.Println("unknown command")
        }
    }
}

func main() {
    baseURL := flag.String("url", "http://localhost:8080", "Base URL")
    flag.Parse()

    client := NewHTTPClient(*baseURL, 30*time.Second)
    cli := NewCLI(client)

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

    go func() {
        <-sigChan
        cancel()
    }()

    if err := cli.Run(ctx); err != nil {
        log.Fatal(err)
    }
}
"""},
    {"domain": "code", "text": """
export interface RequestContext {
  userId: string;
  correlationId: string;
  timestamp: number;
  userAgent: string;
  ipAddress: string;
}

export interface PageRequest {
  page: number;
  limit: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface SearchQuery {
  q: string;
  filters?: Record<string, unknown>;
  facets?: string[];
  boost?: Record<string, number>;
}

class QueryBuilder {
  private filters: Map<string, unknown[]> = new Map();
  private sorts: Array<{ field: string; order: 'asc' | 'desc' }> = [];
  private pagination?: { page: number; limit: number };
  private search?: string;

  addFilter(field: string, value: unknown): QueryBuilder {
    if (!this.filters.has(field)) {
      this.filters.set(field, []);
    }
    this.filters.get(field)!.push(value);
    return this;
  }

  addSort(field: string, order: 'asc' | 'desc' = 'asc'): QueryBuilder {
    this.sorts.push({ field, order });
    return this;
  }

  setPagination(page: number, limit: number): QueryBuilder {
    this.pagination = { page, limit };
    return this;
  }

  setSearch(query: string): QueryBuilder {
    this.search = query;
    return this;
  }

  build(): SearchQuery {
    const filters: Record<string, unknown> = {};
    for (const [key, values] of this.filters) {
      filters[key] = values.length === 1 ? values[0] : values;
    }

    return {
      q: this.search || '*',
      filters,
      facets: Array.from(this.filters.keys()),
    };
  }

  buildElasticsearch(): Record<string, unknown> {
    const query: Record<string, unknown> = {};

    if (this.search) {
      query.query = {
        bool: {
          must: [{ multi_match: { query: this.search, fields: ['*'] } }],
          filter: Array.from(this.filters.entries()).map(([field, values]) => ({
            terms: { [field]: values },
          })),
        },
      };
    } else {
      query.query = {
        bool: {
          filter: Array.from(this.filters.entries()).map(([field, values]) => ({
            terms: { [field]: values },
          })),
        },
      };
    }

    if (this.sorts.length > 0) {
      query.sort = this.sorts.map((s) => ({
        [s.field]: { order: s.order },
      }));
    }

    if (this.pagination) {
      query.from = (this.pagination.page - 1) * this.pagination.limit;
      query.size = this.pagination.limit;
    }

    return query;
  }
}

interface SearchResult<T> {
  total: number;
  page: number;
  limit: number;
  results: T[];
  facets?: Record<string, Array<{ value: string; count: number }>>;
  aggregations?: Record<string, unknown>;
}

class SearchService {
  private baseUrl: string;
  private context: RequestContext;

  constructor(baseUrl: string, context: RequestContext) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.context = context;
  }

  async search<T>(query: SearchQuery): Promise<SearchResult<T>> {
    const url = new URL(`${this.baseUrl}/search`);
    url.searchParams.append('q', query.q);

    if (query.filters) {
      Object.entries(query.filters).forEach(([key, value]) => {
        if (Array.isArray(value)) {
          value.forEach((v) => url.searchParams.append(`filter_${key}`, String(v)));
        } else {
          url.searchParams.append(`filter_${key}`, String(value));
        }
      });
    }

    if (query.facets) {
      query.facets.forEach((f) => url.searchParams.append('facet', f));
    }

    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'X-Correlation-ID': this.context.correlationId,
        'User-Agent': this.context.userAgent,
        'X-User-ID': this.context.userId,
      },
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  async index<T>(doc: T, id: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/documents/${id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': this.context.correlationId,
      },
      body: JSON.stringify(doc),
    });

    if (!response.ok) {
      throw new Error(`Indexing failed: ${response.status}`);
    }
  }

  async delete(id: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/documents/${id}`, {
      method: 'DELETE',
      headers: {
        'X-Correlation-ID': this.context.correlationId,
      },
    });

    if (!response.ok) {
      throw new Error(`Deletion failed: ${response.status}`);
    }
  }
}

export { QueryBuilder, SearchService };
"""},
    {"domain": "code", "text": """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

typedef struct {
    uint8_t* data;
    size_t capacity;
    size_t size;
} ByteBuffer;

typedef struct {
    const char* key;
    void* value;
} HashMap_Entry;

typedef struct {
    HashMap_Entry* entries;
    size_t capacity;
    size_t size;
} HashMap;

ByteBuffer* bytebuffer_new(size_t initial_capacity) {
    ByteBuffer* buf = malloc(sizeof(ByteBuffer));
    if (!buf) return NULL;

    buf->data = malloc(initial_capacity);
    if (!buf->data) {
        free(buf);
        return NULL;
    }

    buf->capacity = initial_capacity;
    buf->size = 0;
    return buf;
}

void bytebuffer_free(ByteBuffer* buf) {
    if (buf) {
        free(buf->data);
        free(buf);
    }
}

int bytebuffer_reserve(ByteBuffer* buf, size_t additional) {
    if (buf->size + additional <= buf->capacity) {
        return 0;
    }

    size_t new_capacity = buf->capacity * 2;
    while (new_capacity < buf->size + additional) {
        new_capacity *= 2;
    }

    uint8_t* new_data = realloc(buf->data, new_capacity);
    if (!new_data) return -1;

    buf->data = new_data;
    buf->capacity = new_capacity;
    return 0;
}

int bytebuffer_append(ByteBuffer* buf, const uint8_t* data, size_t len) {
    if (bytebuffer_reserve(buf, len) != 0) return -1;
    memcpy(buf->data + buf->size, data, len);
    buf->size += len;
    return 0;
}

int bytebuffer_append_string(ByteBuffer* buf, const char* str) {
    size_t len = strlen(str);
    return bytebuffer_append(buf, (const uint8_t*)str, len);
}

uint32_t hash_string(const char* str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

HashMap* hashmap_new(size_t initial_capacity) {
    HashMap* map = malloc(sizeof(HashMap));
    if (!map) return NULL;

    map->entries = calloc(initial_capacity, sizeof(HashMap_Entry));
    if (!map->entries) {
        free(map);
        return NULL;
    }

    map->capacity = initial_capacity;
    map->size = 0;
    return map;
}

void hashmap_free(HashMap* map) {
    if (map) {
        for (size_t i = 0; i < map->capacity; i++) {
            if (map->entries[i].key) {
                free((void*)map->entries[i].key);
            }
        }
        free(map->entries);
        free(map);
    }
}

static size_t hashmap_find_slot(HashMap* map, const char* key) {
    uint32_t hash = hash_string(key);
    size_t index = hash % map->capacity;

    while (map->entries[index].key != NULL) {
        if (strcmp(map->entries[index].key, key) == 0) {
            return index;
        }
        index = (index + 1) % map->capacity;
    }

    return index;
}

int hashmap_set(HashMap* map, const char* key, void* value) {
    if (map->size >= map->capacity / 2) {
        HashMap* new_map = hashmap_new(map->capacity * 2);
        if (!new_map) return -1;

        for (size_t i = 0; i < map->capacity; i++) {
            if (map->entries[i].key) {
                hashmap_set(new_map, map->entries[i].key, map->entries[i].value);
            }
        }

        hashmap_free(map);
        memcpy(map, new_map, sizeof(HashMap));
        free(new_map);
    }

    size_t slot = hashmap_find_slot(map, key);
    if (map->entries[slot].key == NULL) {
        map->entries[slot].key = malloc(strlen(key) + 1);
        if (!map->entries[slot].key) return -1;
        strcpy((char*)map->entries[slot].key, key);
        map->size++;
    }

    map->entries[slot].value = value;
    return 0;
}

void* hashmap_get(HashMap* map, const char* key) {
    if (!map) return NULL;

    uint32_t hash = hash_string(key);
    size_t index = hash % map->capacity;

    while (map->entries[index].key != NULL) {
        if (strcmp(map->entries[index].key, key) == 0) {
            return map->entries[index].value;
        }
        index = (index + 1) % map->capacity;
    }

    return NULL;
}

int hashmap_delete(HashMap* map, const char* key) {
    size_t slot = hashmap_find_slot(map, key);
    if (map->entries[slot].key == NULL) {
        return -1;
    }

    free((void*)map->entries[slot].key);
    map->entries[slot].key = NULL;
    map->entries[slot].value = NULL;
    map->size--;
    return 0;
}
"""},
    {"domain": "code", "text": """
SELECT
    t1.order_id,
    t1.customer_id,
    c.customer_name,
    c.country,
    t1.order_date,
    t1.total_amount,
    COUNT(DISTINCT t2.product_id) as product_count,
    STRING_AGG(DISTINCT p.category, ', ' ORDER BY p.category) as categories,
    ROW_NUMBER() OVER (PARTITION BY c.country ORDER BY t1.total_amount DESC) as rank_in_country,
    LAG(t1.total_amount) OVER (PARTITION BY t1.customer_id ORDER BY t1.order_date) as previous_order_amount,
    CASE
        WHEN t1.total_amount > AVG(t1.total_amount) OVER (PARTITION BY c.country) THEN 'Above Average'
        WHEN t1.total_amount < AVG(t1.total_amount) OVER (PARTITION BY c.country) THEN 'Below Average'
        ELSE 'Average'
    END as amount_category,
    EXTRACT(YEAR FROM t1.order_date) as order_year,
    EXTRACT(MONTH FROM t1.order_date) as order_month,
    DATEDIFF(CURRENT_DATE, t1.order_date) as days_since_order,
    SUM(CASE WHEN oi.quantity > 5 THEN oi.quantity * p.price ELSE 0 END)
        OVER (PARTITION BY t1.order_id) as bulk_order_value
FROM orders t1
INNER JOIN order_items oi ON t1.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id
INNER JOIN customers c ON t1.customer_id = c.customer_id
INNER JOIN categories cat ON p.category_id = cat.category_id
LEFT JOIN order_reviews r ON t1.order_id = r.order_id
WHERE
    t1.order_date >= DATE_TRUNC('year', CURRENT_DATE - INTERVAL '3 years')
    AND c.country IN ('United States', 'Canada', 'United Kingdom', 'Germany', 'France')
    AND t1.order_status NOT IN ('Cancelled', 'Refunded')
    AND p.price > 0
    AND t1.total_amount IS NOT NULL
GROUP BY
    t1.order_id,
    t1.customer_id,
    c.customer_name,
    c.country,
    t1.order_date,
    t1.total_amount,
    p.category
HAVING
    COUNT(DISTINCT t2.product_id) >= 2
    AND SUM(oi.quantity) > 0
ORDER BY
    c.country,
    rank_in_country,
    t1.order_date DESC
LIMIT 10000;

WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', order_date)::DATE as month,
        customer_id,
        COUNT(*) as order_count,
        SUM(total_amount) as monthly_total,
        AVG(total_amount) as avg_order_value
    FROM orders
    WHERE order_status = 'Completed'
    GROUP BY DATE_TRUNC('month', order_date), customer_id
),
customer_metrics AS (
    SELECT
        customer_id,
        COUNT(DISTINCT month) as active_months,
        AVG(monthly_total) as avg_monthly_spend,
        MAX(monthly_total) as peak_month_spend,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY monthly_total) as median_spend
    FROM monthly_sales
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    c.customer_name,
    cm.active_months,
    cm.avg_monthly_spend,
    cm.peak_month_spend,
    ROUND(cm.avg_monthly_spend / NULLIF(cm.active_months, 0), 2) as avg_per_active_month,
    DENSE_RANK() OVER (ORDER BY cm.peak_month_spend DESC) as peak_spend_rank,
    CASE
        WHEN cm.avg_monthly_spend > 1000 THEN 'Premium'
        WHEN cm.avg_monthly_spend > 500 THEN 'Gold'
        WHEN cm.avg_monthly_spend > 100 THEN 'Silver'
        ELSE 'Bronze'
    END as customer_tier
FROM customers c
INNER JOIN customer_metrics cm ON c.customer_id = cm.customer_id
WHERE cm.active_months >= 3
ORDER BY cm.avg_monthly_spend DESC, c.customer_name;
"""},
    {"domain": "code", "text": """
#!/bin/bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="${SCRIPT_DIR}/logs"
readonly DATA_DIR="${SCRIPT_DIR}/data"
readonly BACKUP_DIR="${SCRIPT_DIR}/backups"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly LOG_FILE="${LOG_DIR}/deploy_${TIMESTAMP}.log"

log() {
    local level="$1"
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "$LOG_FILE"
}

initialize() {
    mkdir -p "$LOG_DIR" "$DATA_DIR" "$BACKUP_DIR"
    log "INFO" "Starting deployment process"
    log "INFO" "Script directory: $SCRIPT_DIR"
}

validate_environment() {
    log "INFO" "Validating environment..."

    local required_commands=("git" "docker" "docker-compose" "python3" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR" "Required command not found: $cmd"
            return 1
        fi
    done

    if [ ! -f "${SCRIPT_DIR}/.env" ]; then
        log "ERROR" "Missing .env file"
        return 1
    fi

    log "INFO" "Environment validation passed"
}

backup_current() {
    log "INFO" "Creating backup..."

    if [ -d "${SCRIPT_DIR}/production" ]; then
        tar -czf "${BACKUP_DIR}/production_${TIMESTAMP}.tar.gz" \
            -C "${SCRIPT_DIR}" production \
            --exclude='node_modules' \
            --exclude='.git'
        log "INFO" "Backup created: production_${TIMESTAMP}.tar.gz"
    fi
}

build_services() {
    log "INFO" "Building Docker services..."

    cd "$SCRIPT_DIR"
    docker-compose build --no-cache || {
        log "ERROR" "Docker build failed"
        return 1
    }

    log "INFO" "Docker services built successfully"
}

run_tests() {
    log "INFO" "Running test suite..."

    cd "$SCRIPT_DIR"
    python3 -m pytest tests/ -v --cov=app --cov-report=term-missing || {
        log "ERROR" "Test suite failed"
        return 1
    }

    log "INFO" "All tests passed"
}

migrate_database() {
    log "INFO" "Running database migrations..."

    cd "$SCRIPT_DIR"
    python3 -c "from app.migrations import run_migrations; run_migrations()" || {
        log "ERROR" "Database migration failed"
        return 1
    }

    log "INFO" "Database migrations completed"
}

start_services() {
    log "INFO" "Starting services..."

    cd "$SCRIPT_DIR"
    docker-compose up -d || {
        log "ERROR" "Failed to start services"
        return 1
    }

    sleep 5

    if ! docker-compose ps | grep -q "healthy"; then
        log "WARN" "Some services may not be healthy"
    fi

    log "INFO" "Services started"
}

health_check() {
    log "INFO" "Performing health checks..."

    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log "INFO" "Health check passed"
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 2
    done

    log "ERROR" "Health check failed after $max_attempts attempts"
    return 1
}

rollback() {
    log "ERROR" "Deployment failed, rolling back..."

    cd "$SCRIPT_DIR"
    docker-compose down

    if [ -f "${BACKUP_DIR}/production_${TIMESTAMP}.tar.gz" ]; then
        tar -xzf "${BACKUP_DIR}/production_${TIMESTAMP}.tar.gz" -C "$SCRIPT_DIR"
        log "INFO" "Restored from backup"
    fi

    docker-compose up -d
    log "INFO" "Rollback completed"
}

cleanup() {
    log "INFO" "Cleaning up temporary files..."
    find "${LOG_DIR}" -type f -mtime +7 -delete
    log "INFO" "Cleanup completed"
}

main() {
    initialize

    if ! validate_environment; then
        log "ERROR" "Environment validation failed"
        exit 1
    fi

    trap rollback ERR

    backup_current
    build_services
    run_tests
    migrate_database
    start_services

    if ! health_check; then
        exit 1
    fi

    cleanup
    log "INFO" "Deployment completed successfully"
}

main "$@"
"""},
    {"domain": "code", "text": """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

class BinarySearchTree:
    class Node:
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = self.Node(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = self.Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = self.Node(value)
            else:
                self._insert_recursive(node.right, value)

    def search(self, value):
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def inorder_traversal(self):
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        if node is not None:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_vertex(self, vertex):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self, vertex1, vertex2, weight=1):
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        self.adjacency_list[vertex1].append((vertex2, weight))

    def breadth_first_search(self, start):
        visited = set()
        queue = [start]
        result = []

        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                for neighbor, _ in self.adjacency_list.get(vertex, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

        return result

    def depth_first_search(self, start):
        visited = set()
        result = []
        self._dfs_recursive(start, visited, result)
        return result

    def _dfs_recursive(self, vertex, visited, result):
        visited.add(vertex)
        result.append(vertex)
        for neighbor, _ in self.adjacency_list.get(vertex, []):
            if neighbor not in visited:
                self._dfs_recursive(neighbor, visited, result)

    def dijkstra(self, start):
        distances = {vertex: float('inf') for vertex in self.adjacency_list}
        distances[start] = 0
        unvisited = set(self.adjacency_list.keys())

        while unvisited:
            current = min(unvisited, key=lambda v: distances[v])
            if distances[current] == float('inf'):
                break

            for neighbor, weight in self.adjacency_list[current]:
                if neighbor in unvisited:
                    new_distance = distances[current] + weight
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance

            unvisited.remove(current)

        return distances
"""},
    {"domain": "math", "text": """
The fundamental theorem of calculus establishes the relationship between differentiation and integration, two central operations in calculus. Specifically, it states that if a function f is continuous on a closed interval from a to b and F is an antiderivative of f on that interval, then the definite integral of f from a to b equals F evaluated at b minus F evaluated at a. Mathematically, we express this as the integral from a to b of f of x dx equals F of b minus F of a. This remarkable result demonstrates that differentiation and integration are essentially inverse operations. The theorem has two main parts: the first part tells us that if we integrate the derivative of a continuous function, we recover the original function up to a constant. The second part provides a practical method for computing definite integrals by finding antiderivatives rather than using limit-based definitions. Consider the function f of x equals three x squared. To find the area under this curve from x equals one to x equals three, we first find an antiderivative, which is F of x equals x cubed. Then the definite integral equals F of three minus F of one, which equals twenty-seven minus one, equals twenty-six. This approach is far more efficient than computing limits of Riemann sums. The fundamental theorem also justifies the common notation where we write the indefinite integral of f of x dx to represent the general antiderivative F of x plus a constant C. Applications of this theorem permeate mathematics, physics, and engineering, from calculating work done by forces to finding center of mass to solving differential equations that model real-world phenomena.

The proof of the first part rests on the mean value theorem for integrals. Define the accumulation function capital A of x as the integral from a to x of f of t dt. To compute the derivative of capital A at a point x, form the difference quotient: capital A of x plus h minus capital A of x, all divided by h. The numerator is precisely the integral from x to x plus h of f of t dt, because the accumulated area from a to x plus h minus the accumulated area from a to x leaves exactly the sliver between x and x plus h. The mean value theorem guarantees that this sliver equals h times f of c for some point c lying between x and x plus h. Dividing by h leaves f of c. As h shrinks toward zero, the point c is squeezed toward x, and continuity of f forces f of c to approach f of x. Therefore the derivative of capital A at x is f of x, which is the assertion that the accumulation function is an antiderivative of the integrand.

Continuity is not a decorative hypothesis. Consider the signum function, which takes the value negative one for negative inputs, zero at the origin, and positive one for positive inputs. Its accumulation function is the absolute value of x minus a constant, and that function has no derivative at the origin, precisely where the integrand jumps. Integrability is a weaker requirement than continuity, so a bounded function with finitely many jump discontinuities still possesses an accumulation function, but that function is merely continuous rather than differentiable at the jumps. The Lebesgue theory later repairs this by characterizing exactly which functions arise as derivatives of their own integrals, and the answer involves absolute continuity rather than mere continuity.

The theorem also powers the two workhorse techniques of integration. Substitution is the chain rule read backwards: if u equals g of x, then the integral of f of g of x times g prime of x dx becomes the integral of f of u du, and when the integral is definite the limits themselves transform, so that x running from a to b corresponds to u running from g of a to g of b. Integration by parts is the product rule read backwards: the integral of u dv equals u times v minus the integral of v du. Applied to the integral of x times e to the x dx, choosing u equals x and dv equals e to the x dx yields x times e to the x minus the integral of e to the x dx, which equals x times e to the x minus e to the x plus a constant. Differentiating that answer recovers the original integrand, confirming the result.

Improper integrals extend the framework to unbounded intervals and unbounded integrands by taking limits of ordinary definite integrals. The integral from one to infinity of one over x to the p dx converges precisely when p exceeds one, diverging otherwise, and this single fact underlies the integral test for series convergence. When an antiderivative cannot be expressed in elementary terms, as happens for e to the negative x squared, numerical quadrature takes over. The trapezoidal rule approximates the integrand by straight segments and incurs error proportional to the square of the step size, while Simpson's rule fits parabolas through consecutive triples of points and achieves error proportional to the fourth power of the step size. Gaussian quadrature does better still by choosing both the sample points and their weights to integrate polynomials of high degree exactly, achieving remarkable accuracy from surprisingly few evaluations of the integrand.
"""},
    {"domain": "math", "text": """
Linear algebra provides the mathematical framework for understanding systems of linear equations, vector spaces, and linear transformations. When we write a system of equations in matrix form as A times x equals b, where A is an m by n matrix, x is a vector of unknowns, and b is a vector of constants, we can apply matrix operations to solve for x. If the matrix A is square and invertible, then x equals A inverse times b. The invertibility of a matrix depends on whether its determinant is nonzero. The determinant is a scalar value computed from the entries of a square matrix that captures important geometric information. For a two by two matrix with entries a, b, c, d in row-major order, the determinant equals ad minus bc. For larger matrices, we compute determinants using cofactor expansion or other methods. Eigenvalues and eigenvectors are fundamental concepts where for a square matrix A and a nonzero vector v, if A times v equals lambda times v for some scalar lambda, then lambda is an eigenvalue and v is a corresponding eigenvector. These concepts reveal the fundamental directions in which a linear transformation acts by simple scalar multiplication. The characteristic polynomial, obtained by computing the determinant of A minus lambda times I where I is the identity matrix, has roots equal to the eigenvalues. Understanding eigendecomposition allows us to understand the behavior of iterative processes and differential equations. For example, the PageRank algorithm relies on finding the eigenvector corresponding to the largest eigenvalue of a transition matrix. Matrix diagonalization, when possible, allows us to express A equals P times D times P inverse where D is diagonal, enabling efficient computation of powers like A to the hundredth power by computing P times D to the hundredth times P inverse instead.

Not every matrix is diagonalizable. A matrix fails to diagonalize when some eigenvalue's geometric multiplicity, the dimension of its eigenspace, falls short of its algebraic multiplicity, its multiplicity as a root of the characteristic polynomial. The two by two matrix with ones on the diagonal and a single one in the upper right corner illustrates this: its only eigenvalue is one with algebraic multiplicity two, yet its eigenspace is one dimensional. Jordan normal form repairs the deficiency by permitting blocks with ones on the superdiagonal, and every square matrix over the complex numbers is similar to such a form. For practical computation the Jordan form is numerically treacherous, since arbitrarily small perturbations can split a repeated eigenvalue and change the block structure discontinuously.

The singular value decomposition sidesteps these difficulties entirely and applies to every matrix, square or not. It factors A as U times Sigma times V transpose, where U and V carry orthonormal columns and Sigma is a nonnegative diagonal matrix of singular values, conventionally listed in decreasing order. The singular values are the square roots of the eigenvalues of A transpose times A, a symmetric positive semidefinite matrix whose spectral theory is unusually well behaved. Geometrically the decomposition says that every linear map is a rotation, followed by an axis-aligned stretch, followed by another rotation. Truncating the decomposition after the largest k singular values yields the best rank k approximation to A measured in either the Frobenius or the spectral norm, a result known as the Eckart-Young theorem, and this single fact underpins principal component analysis, latent semantic indexing, and lossy image compression.

Orthogonality organizes much of the subject. Two vectors are orthogonal when their inner product vanishes, and an orthonormal basis makes coordinate computation trivial, since the coefficient of each basis vector is simply the inner product with it. The Gram-Schmidt process converts any linearly independent collection into an orthonormal one by successively subtracting off projections onto the vectors already produced, and performing this on the columns of A yields the QR factorization, A equals Q times R with Q orthonormal and R upper triangular. In floating point arithmetic the classical Gram-Schmidt loses orthogonality badly, so implementations use the modified variant or Householder reflections instead.

Least squares problems arise whenever a system has more equations than unknowns and no exact solution exists. Minimizing the squared length of the residual b minus A times x leads to the normal equations, A transpose times A times x equals A transpose times b. Solving these directly squares the condition number of A and can destroy accuracy, so the QR factorization is preferred, reducing the problem to back substitution against the triangular factor R. The condition number itself, defined as the ratio of the largest singular value to the smallest, quantifies how much a relative perturbation in the data can be amplified in the solution. A matrix with condition number of order ten to the eighth will lose roughly eight decimal digits of accuracy in double precision arithmetic, which is why ill-conditioned systems demand regularization, such as adding a small multiple of the identity to A transpose times A, the technique known as ridge regression or Tikhonov regularization.

The four fundamental subspaces tie these threads together. The column space and the null space of A transpose are orthogonal complements within the output space, while the row space and the null space of A are orthogonal complements within the input space. The rank-nullity theorem states that the rank plus the nullity equals the number of columns, and the singular value decomposition exhibits orthonormal bases for all four subspaces simultaneously, which is why it serves as the definitive computational tool for questions of rank, range, and approximate linear dependence.
"""},
    {"domain": "math", "text": """
Probability theory provides the mathematical foundation for quantifying uncertainty and randomness. A sample space Omega is the set of all possible outcomes of a random experiment. An event is a subset of the sample space, and a probability measure P assigns to each event a number between zero and one inclusive such that P of Omega equals one. For discrete sample spaces with equally likely outcomes, the probability of an event E equals the number of favorable outcomes divided by the total number of possible outcomes. For continuous sample spaces, we use probability density functions where the probability that a random variable X falls in an interval from a to b equals the integral from a to b of the probability density function f of x dx. The cumulative distribution function F of x equals the probability that X is less than or equal to x. Expected value, also called the mean, represents the long-run average value of a random variable. For a discrete random variable, E of X equals the sum over all possible values of x times the probability that X equals x. For a continuous random variable, E of X equals the integral from negative infinity to infinity of x times f of x dx. Variance measures the spread around the mean and is defined as E of the quantity X minus E of X all squared, which simplifies to E of X squared minus E of X squared. The standard deviation is the square root of variance. Common probability distributions include the normal distribution with probability density function proportional to e to the negative x minus mu squared over two sigma squared, the binomial distribution governing the number of successes in n independent trials each with success probability p, and the Poisson distribution useful for modeling count data with rate parameter lambda. Independence of events means P of A and B equals P of A times P of B, and conditional probability P of A given B equals P of A and B divided by P of B.

Rearranging the definition of conditional probability produces Bayes' theorem: P of A given B equals P of B given A times P of A, divided by P of B. Combined with the law of total probability, which expands the denominator as a sum over a partition of the sample space, this formula converts forward knowledge into backward inference. A standard illustration concerns medical screening. Suppose a disease afflicts one person in a thousand, and a test detects it with sensitivity ninety-nine percent while producing a false positive in five percent of healthy people. Given a positive result, the posterior probability of disease is the product of one in a thousand and ninety-nine hundredths, divided by that same product plus the product of nine hundred ninety-nine thousandths and five hundredths. The arithmetic yields roughly two percent. Most positives are false, not because the test is poor but because the healthy population is so much larger that even a small false positive rate produces more false alarms than true detections. This base rate effect is among the most reliably misjudged facts in applied reasoning.

Two limit theorems govern the behavior of averages. The law of large numbers states that the sample mean of independent identically distributed random variables with finite expectation converges to the true mean as the sample size grows. The weak form asserts convergence in probability, meaning the chance of deviating by more than any fixed amount tends to zero; the strong form asserts that the sequence of sample means converges to the mean with probability one. The central limit theorem is sharper and more surprising: for variables with finite variance sigma squared, the quantity formed by subtracting the mean from the sample average and dividing by sigma over the square root of n converges in distribution to a standard normal, regardless of the shape of the underlying distribution. This universality explains why the bell curve appears throughout the empirical sciences, and it justifies confidence intervals of the form sample mean plus or minus roughly two standard errors.

Generating functions provide an algebraic handle on distributions. The moment generating function, defined as the expectation of e to the t X, encodes every moment in its derivatives at the origin, and the moment generating function of a sum of independent variables is the product of the individual ones, converting convolution into multiplication. When the moment generating function fails to exist, as it does for heavy-tailed distributions such as the Cauchy, the characteristic function defined with an imaginary exponent always exists and serves the same purpose. The Cauchy distribution is instructive as a cautionary case: it has no finite mean, its sample average has the same distribution as a single observation no matter how many samples are drawn, and the central limit theorem simply does not apply.

Covariance measures joint variation, defined as the expectation of the product of the two centered variables, and the correlation coefficient normalizes it to lie between negative one and one. Zero correlation does not imply independence; a variable uniformly distributed on the interval from negative one to one is uncorrelated with its own square yet is obviously dependent on it. Concentration inequalities bound deviations without requiring full distributional knowledge. Markov's inequality bounds the probability that a nonnegative variable exceeds a threshold by its mean divided by that threshold. Chebyshev's inequality, obtained by applying Markov to the squared deviation, bounds the probability of straying more than k standard deviations from the mean by one over k squared. Hoeffding's inequality does far better for bounded variables, giving a bound that decays exponentially in the sample size, which is why it underlies much of statistical learning theory. Markov chains extend the framework to dependent sequences, where the future depends on the past only through the present state, and under mild conditions such chains converge to a unique stationary distribution independent of where they started.
"""},
    {"domain": "math", "text": """
Fourier analysis decomposes periodic functions into sums of sine and cosine functions, revealing the frequency components present in a signal. The Fourier series of a periodic function f with period two L on the interval from negative L to L is the infinite series consisting of a zero over two plus the sum from n equals one to infinity of a n cosine of n pi x over L plus b n sine of n pi x over L. The coefficients are computed using orthogonality of trigonometric functions: a n equals one over L times the integral from negative L to L of f of x cosine of n pi x over L dx, and similarly for b n with sine. The remarkable result is that under reasonable conditions on f, this infinite series converges to f of x. For non-periodic functions defined on the entire real line, the Fourier transform extends this idea by defining the Fourier transform of f as the integral from negative infinity to infinity of f of x e to the negative i omega x dx, where omega is the frequency variable and i is the imaginary unit. The inverse Fourier transform recovers f from its Fourier transform. This pair of transformations is fundamental in signal processing, where Fourier analysis reveals what frequencies are present in a time-domain signal. The power spectrum, defined as the absolute value squared of the Fourier transform, shows how much power resides at each frequency. The convolution theorem states that convolution in one domain corresponds to pointwise multiplication in the Fourier domain, which motivates using Fast Fourier Transform algorithms to efficiently compute convolutions. Parseval's theorem relates the energy in the time domain to energy in the frequency domain: the integral of f of x squared dx equals one over two pi times the integral of the absolute value squared of the Fourier transform of f of omega d omega. These tools enable compression, noise filtering, and frequency-based analysis across engineering and science.

Digital computation requires a discrete counterpart. The discrete Fourier transform takes a finite sequence of N samples and produces N complex coefficients, each formed by summing the samples against powers of a primitive N-th root of unity. Computed naively this costs N squared multiplications, which becomes prohibitive for long signals. The Cooley-Tukey algorithm exploits the recursive structure of the roots of unity, splitting the sum into even-indexed and odd-indexed halves, each of which is itself a transform of length N over two. Recursion yields a cost proportional to N times the logarithm of N. For a sequence of one million samples this reduces the operation count from roughly ten to the twelfth to roughly two times ten to the seventh, a factor of fifty thousand, and that single algorithmic improvement made real-time digital signal processing feasible.

Sampling connects the continuous and discrete worlds. The Nyquist-Shannon theorem states that a signal containing no frequencies above B hertz is completely determined by samples taken at any rate exceeding two B samples per second, and the original signal can be reconstructed exactly by interpolating with sinc functions. Sampling too slowly causes aliasing, in which high frequencies masquerade as low ones because they are indistinguishable at the available sample points. This is why a wagon wheel filmed at twenty-four frames per second can appear to rotate backwards, and why analog anti-aliasing filters must precede any analog-to-digital converter. The mathematics is unforgiving: information lost to aliasing cannot be recovered afterward by any amount of processing.

Finite observation windows introduce their own artifacts. Truncating an infinite signal to a finite record is equivalent to multiplying by a rectangular window, and by the convolution theorem this convolves the true spectrum with the transform of the rectangle, a sinc function whose sidelobes decay only slowly. Energy from a strong frequency component therefore leaks into neighboring bins and can bury weaker components entirely. Tapered windows such as the Hann, Hamming, and Blackman windows reduce sidelobe leakage at the cost of widening the main lobe, trading frequency resolution for dynamic range, and the choice among them is dictated by whether the analyst needs to separate nearby frequencies or detect faint ones beside loud ones.

A related pathology afflicts Fourier series of discontinuous functions. Near a jump, the partial sums overshoot the true value by approximately nine percent of the jump height, and this overshoot does not diminish as more terms are added; it merely narrows. The Gibbs phenomenon is not a numerical error but a genuine feature of pointwise convergence failing to be uniform, and it explains the ringing artifacts visible near sharp edges in JPEG images compressed at aggressive quality settings.

Fourier analysis has a structural limitation: it reports which frequencies are present but not when they occurred, since each coefficient integrates over the entire record. The short-time Fourier transform addresses this by sliding a window along the signal and transforming each segment, producing a spectrogram, but the window length imposes a fixed compromise between temporal and spectral precision. This trade-off is not an engineering shortcoming; it is a theorem. The time-bandwidth product of any signal is bounded below, the same inequality that appears in quantum mechanics as the Heisenberg uncertainty principle, since position and momentum are Fourier conjugates in exactly the manner of time and frequency. Wavelet transforms respond by using basis functions that are localized in both domains and scaled rather than merely translated, giving fine temporal resolution at high frequencies and fine spectral resolution at low ones. This adaptive tiling of the time-frequency plane suits transient signals such as seismic traces, electrocardiograms, and image edges, and it forms the mathematical basis of the JPEG 2000 compression standard.
"""},
    {"domain": "math", "text": """
Differential equations are equations relating a function to its derivatives and are essential for modeling phenomena that change continuously. An ordinary differential equation involves derivatives with respect to a single independent variable, typically time. A first-order linear ODE has the form dy over dt plus p of t times y equals q of t. To solve this, we multiply by an integrating factor, which is e to the integral of p of t dt. A separable equation has the form dy over dt equals f of t times g of y, and we can separate variables to get dy over g of y equals f of t dt, then integrate both sides. Higher-order equations involve derivatives of higher order. A second-order linear ODE with constant coefficients has the form a times d squared y over dt squared plus b times dy over dt plus c times y equals f of t. The homogeneous equation with f of t equals zero has a characteristic equation a times r squared plus b times r plus c equals zero. If the roots are r one and r two, the general solution to the homogeneous equation is C one times e to the r one t plus C two times e to the r two t when the roots are distinct and real. The method of undetermined coefficients or variation of parameters gives particular solutions for the non-homogeneous case. Systems of coupled differential equations describe interactions between multiple variables. For example, predator-prey dynamics are modeled by Lotka-Volterra equations: dx over dt equals alpha times x minus beta times x times y, and dy over dt equals negative gamma times y plus delta times x times y, where x and y represent populations. Stability analysis using linearization around equilibrium points and analysis of the Jacobian matrix determines whether equilibria are stable, unstable, or saddle points. Numerical methods like Runge-Kutta schemes are essential for solving differential equations when analytical solutions are not available.

Before seeking a solution it is worth asking whether one exists and whether it is unique. The Picard-Lindelöf theorem answers both: if the right-hand side of the equation is continuous in time and Lipschitz continuous in the dependent variable on some rectangle around the initial condition, then a unique solution exists on some interval containing the initial time. The Lipschitz condition cannot be dropped. The equation dy over dt equals the cube root of y squared, started from zero, admits both the identically zero solution and a solution that departs from zero after any delay whatsoever, because the cube root has unbounded slope at the origin. Uniqueness fails, and with it any hope of prediction. Even when uniqueness holds, solutions may exist only locally: dy over dt equals y squared with initial value one produces y equals one over one minus t, which escapes to infinity at time one despite the equation being perfectly well behaved everywhere.

The damped harmonic oscillator repays detailed study because it exhibits every qualitative behavior of second-order linear systems. Writing m times the second derivative of x plus c times the first derivative plus k times x equals zero, the characteristic roots are determined by the discriminant c squared minus four m k. When the discriminant is positive the roots are real and distinct, the system is overdamped, and it returns to equilibrium without crossing it. When the discriminant vanishes the roots coincide, the system is critically damped, and the general solution acquires a factor of t multiplying the second exponential; this case returns to equilibrium fastest, which is why door closers and galvanometer needles are tuned to it. When the discriminant is negative the roots are complex conjugates, and the solution is an exponentially decaying envelope multiplying a sinusoid, producing the familiar ringing of an underdamped system. Driving such an oscillator at its natural frequency produces resonance, in which the particular solution grows without bound in the undamped limit, a phenomenon responsible for both the utility of radio tuners and the destruction of poorly designed bridges.

The Laplace transform converts linear constant-coefficient equations into algebra. Defined as the integral from zero to infinity of e to the negative s t times f of t dt, it turns differentiation into multiplication by s, with initial conditions entering automatically as additive terms. An initial value problem becomes a rational function of s, which is decomposed into partial fractions and inverted term by term against a table of standard transforms. The method handles discontinuous forcing, such as a step voltage applied to a circuit, and impulsive forcing represented by the Dirac delta, with no special treatment, which is why it dominates control engineering. The transfer function, the ratio of output transform to input transform, encodes the entire input-output behavior of a linear system, and the locations of its poles in the complex plane determine stability: poles in the left half-plane give decaying responses, poles on the imaginary axis give sustained oscillation, and poles in the right half-plane give exponential blowup.

For nonlinear systems, qualitative analysis often substitutes for explicit solution. The phase plane plots one dependent variable against another, with trajectories tracing the system's evolution. Equilibria are located where all derivatives vanish, and linearizing about each one reduces the local question to the eigenvalues of the Jacobian matrix. Eigenvalues with negative real parts indicate a stable node or spiral; mixed signs indicate a saddle point, unstable but with a stable direction; purely imaginary eigenvalues indicate a center in the linearization, where the nonlinear terms decide the outcome and no conclusion can be drawn from the linear analysis alone. Lyapunov functions offer a complementary approach, establishing stability by exhibiting an energy-like quantity that decreases along trajectories, and they work even where linearization is inconclusive.

Numerical integration introduces its own subtleties. Euler's method advances by a single slope evaluation and accumulates error proportional to the step size, while fourth-order Runge-Kutta combines four evaluations per step and accumulates error proportional to the fourth power, making it vastly more efficient for smooth problems. Stiff systems, in which processes evolve on wildly separated timescales, defeat explicit methods entirely: stability rather than accuracy dictates a punishingly small step size. Implicit schemes such as backward Euler or the backward differentiation formulas remain stable at large steps but require solving a nonlinear system at each step, a cost that is nonetheless worth paying for chemical kinetics and circuit simulation.
"""},
    {"domain": "math", "text": """
Complex analysis studies functions of complex variables and reveals surprising connections between analysis and geometry. The complex plane represents numbers z equals x plus iy where x and y are real and i is the imaginary unit with i squared equals negative one. A function f from complex numbers to complex numbers can be written as f of z equals u of x y plus i times v of x y where u and v are real-valued functions. Such a function is analytic or holomorphic in a region if it is differentiable everywhere in that region, which is possible only if the Cauchy-Riemann equations are satisfied: the partial derivative of u with respect to x equals the partial derivative of v with respect to y, and the partial derivative of u with respect to y equals negative the partial derivative of v with respect to x. A remarkable result is that if f is analytic on and inside a closed contour C, then the contour integral of f of z dz around C equals zero. More generally, the residue theorem states that the contour integral of f of z dz equals two pi i times the sum of residues at poles inside the contour. The residue at a simple pole z equals z zero is the limit as z approaches z zero of z minus z zero times f of z. These results allow us to compute difficult real integrals by extending them to the complex plane and using residues. Power series representations of analytic functions show that locally every analytic function equals a convergent power series. The Laurent series generalizes power series to allow negative powers of z minus z zero and converges in an annulus around a singularity. Conformal mappings, which are analytic functions whose derivatives are nonzero, preserve angles and are used in engineering applications to transform complicated domains into simpler ones for solving physical problems like fluid flow and electrostatics.

Analyticity is a far more restrictive condition than real differentiability, and the consequences are correspondingly dramatic. A function differentiable once in a complex neighborhood is automatically differentiable infinitely often, a statement with no analogue on the real line, where a function can have a first derivative but no second. Cauchy's integral formula supplies the mechanism: the value of an analytic function at an interior point equals a contour integral of the function divided by z minus that point, and differentiating under the integral sign as often as one pleases is legitimate because the integrand's dependence on the point is smooth. The same formula yields Cauchy's estimates, bounding the derivatives at a point in terms of the maximum of the function on a surrounding circle.

From those estimates flows Liouville's theorem: a function analytic on the entire complex plane and bounded must be constant. The proof is three lines, yet it immediately delivers the fundamental theorem of algebra. If a nonconstant polynomial had no root, its reciprocal would be entire, and since a polynomial grows without bound at infinity the reciprocal would be bounded, forcing it to be constant and contradicting the assumption. Every nonconstant polynomial with complex coefficients therefore has a root, and by induction it factors completely into linear factors, a purely algebraic conclusion reached by analytic means.

The maximum modulus principle asserts that a nonconstant analytic function attains no local maximum of its absolute value in the interior of its domain; the maximum is always achieved on the boundary. Physically this says that the temperature in a steady-state heat distribution, or the electrostatic potential in a charge-free region, cannot peak in the interior, which is why lightning strikes the tips of conductors and why field strength concentrates at sharp corners.

Multivalued functions demand care. The complex logarithm satisfies e to the log z equals z, but since the exponential is periodic with period two pi i, the logarithm is determined only up to that additive ambiguity. Fixing a single-valued branch requires excising a branch cut, conventionally the negative real axis, across which the function jumps discontinuously. Square roots present the same difficulty, and following a branch continuously around a circle enclosing the origin returns the opposite sign, which is why the origin is called a branch point. Riemann's response was to abandon the plane and construct a surface on which the function becomes genuinely single-valued, with sheets glued along the cuts. The square root lives naturally on a two-sheeted surface, the logarithm on an infinite spiral, and this geometric reconception transformed the subject.

Analytic continuation extends a function beyond the disc where its defining series converges. If two analytic functions agree on any set possessing a limit point within a connected domain, they agree everywhere on that domain, so the extension, when it exists, is unique. The Riemann zeta function is defined initially by the sum over positive integers of one over n to the s, converging only when the real part of s exceeds one, yet analytic continuation extends it to the whole plane apart from a simple pole at one. Its zeros in the critical strip encode the distribution of prime numbers, and the conjecture that all nontrivial zeros lie on the line with real part one half remains the most consequential open problem in mathematics. The gamma function similarly continues the factorial from the positive integers to the complex plane, satisfying gamma of z plus one equals z times gamma of z, with simple poles at the nonpositive integers.

The argument principle counts zeros and poles by contour integration: the integral of the logarithmic derivative around a closed curve equals two pi i times the number of zeros minus the number of poles inside, each counted with multiplicity. Rouché's theorem builds on it, asserting that if one function dominates another in magnitude along a contour, their sum has the same number of zeros inside as the dominant function alone, which provides a practical technique for locating roots of polynomials and for proving stability criteria in control theory.
"""},
    {"domain": "math", "text": """
Abstract algebra studies algebraic structures like groups, rings, and fields. A group G is a set with a binary operation that satisfies closure, associativity, the existence of an identity element, and the existence of inverses. For example, the integers under addition form an infinite cyclic group generated by one, and the set of nonzero real numbers under multiplication forms a group. A subgroup H of a group G is a subset that is itself a group under the same operation. Lagrange's theorem states that the order of a subgroup divides the order of the finite group. A ring R is a set with addition and multiplication operations where R is an abelian group under addition and multiplication is associative with a distributive property connecting addition and multiplication. A field F is a commutative ring where every nonzero element has a multiplicative inverse. The real numbers and complex numbers are familiar fields. A vector space V over a field F is a set where elements can be added and scaled by field elements, with several axioms ensuring this structure behaves predictably. Basis vectors for a vector space are linearly independent vectors that span the space, and any vector can be uniquely expressed as a linear combination of basis vectors. Linear algebra, in this abstract setting, studies linear transformations between vector spaces, which are functions T such that T of a times u plus b times v equals a times T of u plus b times T of v. The rank-nullity theorem states that the dimension of the domain equals the dimension of the image plus the dimension of the kernel. Polynomial rings with coefficients in a field are important examples of rings, and factorization theory studies how polynomials decompose into irreducible factors, analogous to prime factorization for integers.

The organizing idea of group theory is the homomorphism, a map between groups preserving the operation. Its kernel, the set of elements sent to the identity, is never merely a subgroup but a normal one, meaning it is invariant under conjugation by every element of the group. Normality is exactly the condition permitting the set of cosets to inherit a group structure, producing the quotient group. The first isomorphism theorem then states that the image of a homomorphism is isomorphic to the domain modulo the kernel, which reduces the study of maps to the study of normal subgroups and provides the template for analogous theorems about rings, modules, and vector spaces.

Symmetric groups occupy a central position because Cayley's theorem shows that every finite group embeds in one: a group of order n is isomorphic to a subgroup of the permutations of its own n elements, acting by left multiplication. The symmetric group on n letters contains the alternating group of even permutations as a normal subgroup of index two, and for n at least five the alternating group is simple, possessing no normal subgroups apart from itself and the trivial one. That simplicity is not a curiosity. It is the obstruction that makes the general quintic equation unsolvable by radicals.

Galois theory establishes this by placing field extensions and groups in exact correspondence. Adjoining the roots of a polynomial to its coefficient field produces a splitting field, and the automorphisms of that field fixing the base form the Galois group. The fundamental theorem of Galois theory asserts an inclusion-reversing bijection between intermediate fields and subgroups of the Galois group, with normal subgroups corresponding precisely to extensions that are themselves normal. Solvability by radicals translates into the existence of a chain of subgroups with abelian quotients, a property called solvability of the group. Since the alternating group on five letters is simple and nonabelian, the symmetric group on five letters is not solvable, and therefore no formula in radicals can express the roots of a general quintic. Three classical construction problems fall to the same machinery: doubling the cube and trisecting a general angle require solving cubics whose roots generate degree three extensions, while straightedge and compass constructions can only produce extensions of degree a power of two.

Finite fields illustrate how tightly the axioms constrain structure. A finite field has order equal to a prime power, and for each prime power there is exactly one such field up to isomorphism. The field of order p, for p prime, is the integers modulo p; larger fields are built as quotients of polynomial rings by irreducible polynomials. The multiplicative group of any finite field is cyclic, a fact with immediate cryptographic consequence, since the difficulty of inverting exponentiation in that cyclic group is the hardness assumption underlying Diffie-Hellman key exchange. Reed-Solomon error-correcting codes treat data blocks as coefficients of polynomials over a finite field and exploit the fact that a polynomial of degree k is determined by any k plus one of its values, allowing reconstruction of corrupted symbols. These codes protect compact discs, QR codes, and deep space telemetry.

Ideals play for rings the role normal subgroups play for groups. A principal ideal domain is a ring in which every ideal is generated by a single element, and in such rings unique factorization into irreducibles holds. The integers and the polynomial ring over a field are both principal ideal domains, which is why the Euclidean algorithm works identically in each. Not every ring is so well behaved: in the ring of integers adjoined the square root of negative five, the number six factors as two times three and also as one plus the square root of negative five times its conjugate, with all four factors irreducible. Repairing this failure by passing from elements to ideals was the achievement of Dedekind and Kummer, and it launched algebraic number theory.
"""},
    {"domain": "math", "text": """
Calculus of variations concerns itself with finding functions that minimize or maximize certain functionals, which are mappings from a space of functions to the real numbers. A classic problem is to find the path between two points that minimizes arc length, which led to discovering the brachistochrone curve. The Euler-Lagrange equation provides a necessary condition for a function y of x to extremize the functional J of y equals the integral from a to b of L of x y of x dy over dx dx, where L is called the Lagrangian. Taking the variation and setting it to zero yields d over dx of the partial derivative of L with respect to y prime minus the partial derivative of L with respect to y equals zero. For example, in mechanics, the action integral is the integral of kinetic energy minus potential energy over time, and the path taken is the one that extremizes this action, which is Hamilton's principle. Constrained optimization using Lagrange multipliers introduces multiplier functions to enforce constraints. If we want to extremize f subject to the constraint g equals zero, we form the Lagrangian with a multiplier lambda and solve the system of equations obtained by setting derivatives to zero. The Kuhn-Tucker conditions generalize this to inequality constraints. Second variations determine whether a critical point is a minimum, maximum, or saddle point. Functional derivatives generalize partial derivatives to functionals, defined as the limit as epsilon approaches zero of J of y plus epsilon times eta minus J of y over epsilon for any perturbation eta. These concepts extend to fields where we extremize functionals like the Dirichlet energy in potential theory or the Einstein-Hilbert action in general relativity. Numerical methods like finite element analysis discretize these problems to make them computationally tractable.

The brachistochrone deserves working through, since it was the problem that launched the subject when Johann Bernoulli posed it as a public challenge in 1696. A bead slides without friction from a higher point to a lower one under gravity; which shape of wire minimizes the travel time? Conservation of energy gives the speed at depth y as the square root of twice g times y, and the time is the integral of arc length divided by speed, namely the integral of the square root of the quantity one plus y prime squared, divided by the square root of twice g times y, with respect to x. The Lagrangian contains no explicit x dependence, which permits the Beltrami identity: the quantity L minus y prime times the partial derivative of L with respect to y prime is constant along the solution. Carrying out the algebra yields y times the quantity one plus y prime squared equals a constant, whose solution in parametric form is the cycloid, the curve traced by a point on the rim of a rolling circle. The answer is counterintuitive, since the fastest path dips below the straight line, trading extra distance for early speed.

The absence of an explicit independent variable in the Lagrangian, which made that shortcut available, is a special case of a far deeper principle. Noether's theorem states that every continuous symmetry of the action corresponds to a conserved quantity. Invariance under time translation yields conservation of energy; invariance under spatial translation yields conservation of momentum; invariance under rotation yields conservation of angular momentum; and invariance under phase rotation of a complex field yields conservation of electric charge. This single theorem unifies conservation laws that had previously been discovered piecemeal and empirically, and it remains the organizing principle of theoretical physics.

Geodesics generalize straight lines to curved spaces by extremizing arc length, and the Euler-Lagrange equations for the length functional produce the geodesic equation with its Christoffel symbols. On a sphere the geodesics are great circles, which is why intercontinental flight paths appear curved on a flat map. In general relativity the same variational principle, applied to the spacetime metric, produces the trajectories of freely falling bodies, so that planetary orbits become geodesics rather than responses to a force.

Constraints that are themselves integral quantities give rise to isoperimetric problems, the ancestral example being Dido's problem of enclosing maximum area with a fixed perimeter, whose solution is the circle. Such constraints are handled by adjoining them with constant multipliers, and the multiplier acquires physical meaning as a shadow price, measuring how much the optimum improves per unit relaxation of the constraint. In economics it is the marginal value of a resource; in mechanics it is a constraint force.

The Legendre transformation converts the Lagrangian formulation into the Hamiltonian one by replacing velocities with conjugate momenta, yielding a system of first-order equations in phase space rather than second-order equations in configuration space. This reformulation exposes the symplectic geometry of mechanics, supports canonical transformations that simplify problems by changing coordinates, and provides the bridge to quantum mechanics through the correspondence between Poisson brackets and commutators. Optimal control theory extends the framework further: Pontryagin's maximum principle handles problems where the control enters the dynamics and may be bounded, situations where the classical Euler-Lagrange equations do not apply because the optimum lies on the boundary of the admissible set rather than at an interior stationary point.

Direct methods bypass the differential equations entirely. Rather than deriving and solving the Euler-Lagrange equation, the Ritz method expands the unknown function in a finite basis and minimizes the functional over the resulting finite-dimensional space of coefficients, reducing the problem to ordinary calculus. Choosing basis functions with small support produces the finite element method, whose stiffness matrices are sparse and whose convergence is governed by the approximation properties of the chosen elements.
"""},
    {"domain": "prose", "text": """
The old lighthouse keeper climbed the spiral stairs for the last time, his weathered hands following the worn iron rail he had touched thousands of times before. Fifty years of solitude had shaped him into something neither quite human nor fully inanimate, like the lighthouse itself, standing sentinel at the edge of the world. The beam had rotated every night without fail, a rhythm more reliable than his own heartbeat, and now as he reached the top chamber where the great lens caught the dying sunlight, he understood that both he and the light would soon be extinguished by progress. A automated system was being installed by morning; the younger generation had no need for keepers anymore. He looked out at the impossible expanse of water stretching to the horizon, water that had swallowed ships and secrets in equal measure, and thought about the families who had reached safety because of that light. His daughter lived in the city now and barely visited. She had always wanted something different, something beyond these rocks and fog. He held no grudge for that. The world was changing, becoming smaller and faster, and solitary lights in the darkness seemed quaint to those born into a lit world. As the sun sank lower, painting the clouds in shades of amber and rose, he began the ancient ritual one final time, preparing the great lamp with the same care he had shown the first night he arrived as a young man, full of dreams and afraid of silence. When darkness fell completely, that light blazed forth as it always did, calling across the void to lost travelers and wayfarers, a voice without words, speaking only of hope and safe harbor.

He did not sleep. There seemed no point in surrendering the last hours to unconsciousness. Instead he sat in the wooden chair beside the lamp housing, the one his predecessor had built from a broken spar, and listened to the mechanism turn. The clockwork had been converted to electric drive in the sixties, but the bearing still made a particular sound at the top of each rotation, a soft catch like a held breath, and he had learned to hear its absence before any gauge registered a fault. Twice in fifty years that sound had changed and twice he had been at the gearbox with a lamp in his teeth before the beam faltered. Nobody had ever known. Nobody had needed to.

Toward two in the morning the fog came in, as it usually did when the wind dropped, and the beam stopped being a line and became a solid moving wall, sweeping the vapour in slow revolutions. On such nights the light seemed less like a signal than like something alive, breathing out across the water. He thought of the winter of the freighter, the one whose name he still could not say aloud, when he had watched her lights go from three to two to none in the space of eleven minutes and had been able to do precisely nothing but keep his own light turning so the lifeboats would know which way was away from the rocks. Four of them had made it. He had met one of the survivors years later, a Latvian deckhand with a burned hand, who had come up the stairs and said nothing at all for a long while and then simply put that hand on the brass rail and left.

Near dawn he did the things he would not be able to do again. He polished the inner glass with the chamois, though it did not need it. He filled in the log, noting wind, visibility, and the hour, in the same cramped hand he had used since he was twenty-three, and then, after a pause, he wrote a final line recording the transfer of the station to automatic operation, and signed it, and set the pen down beside the book rather than in his pocket, because the pen belonged to the lighthouse and not to him.

The technicians arrived at seven with a van and a great deal of equipment in grey cases. They were polite and young and faintly embarrassed, as people are around something they are in the process of ending. One of them asked whether he had any advice about the site. He considered the question seriously, because it had been asked seriously, and then he told them about the bearing, about the sound at the top of the rotation, and watched the young man write it down in a phone with a small frown, filing away a fact he would never be able to use. They shook his hand. They said the word legend, which meant nothing.

He carried one bag down the spiral, and did not count the steps, and at the bottom he turned and looked up the shaft of the tower into the grey circle of morning at the top. The light was off now, waiting for a dusk it would meet without him. He found that he was not sad in the way he had prepared to be sad. Something had been handed on, imperfectly, in the wrong shape, to people who did not understand what they had received. But it would still burn tonight. On the whole, he decided, walking out along the causeway while the tide came in behind him, that was the part that had always mattered.
"""},
    {"domain": "prose", "text": """
Dr. Amara Okafor had spent fifteen years studying the disappearance of the Lagos songbirds, watching how the city's rapid expansion had silenced what had once been an orchestra of dawn choruses. Her research had led her to back gardens and hidden green spaces, where she documented every remaining species, their shrinking territories marked on maps that broke her heart. She became something of an obsession to her colleagues, this woman who could identify a bird by its shadow, who mourned the extinction of each species as if losing family members. Her apartment was filled with sketches, recordings, photographs of birds already gone to the world, preserved only in her meticulous notes. One morning, while recording in an overgrown lot scheduled for demolition, she encountered an elderly man tending a secret garden among the ruins. He had been maintaining this small paradise for forty years, he explained, hidden from view, saving seeds and raising plants that had been pushed out of the city. Together they began cataloging the bird species that returned when the garden grew. Her research papers began to catch attention not because of her conclusions, but because they seemed to suggest that loss was not entirely inevitable. Developers were contacted, conversations began. The garden became a case study in urban resilience. She thought often of how fragile preservation was, how dependent on passion and stubbornness and luck, on an old man's quiet defiance and her own refusal to accept extinction as inevitable.

The old man's name was Baba Sunday, and he had been a signals technician for the railways before the railways stopped, which explained the meticulousness of his record-keeping. He kept his own notes in three school exercise books, the covers reinforced with tape, recording in a tiny even script the date each plant had been put in, where the seed had come from, whether it had taken. He had no training in botany and did not know most of the scientific names, but he had independently worked out succession planting, companion species, and a rough approximation of a seed bank, storing envelopes of dried seed in a biscuit tin inside a metal drum against the humidity. When Amara showed him the taxonomic keys he was politely uninterested. He wanted to know about the birds.

What he wanted to know, specifically, was why the greenbuls had come back in the fourth year and not before. Amara did not have an answer, and the honesty of saying so seemed to raise rather than lower his estimation of her. Together they worked out a hypothesis over several months: the fruiting shrubs he had planted early took four seasons to produce reliably, and the birds were not responding to the garden as habitat but to it as a food source with a predictable calendar. It was an obvious conclusion in retrospect and a genuinely novel one in the literature for that particular assemblage of species, and when she published it she put his name on the paper. The journal queried the affiliation. She wrote back that his affiliation was the garden, and after some correspondence they printed it that way.

The demolition notice came anyway, eighteen months later, served by a development company that had acquired the parcel through a chain of intermediaries opaque enough to require a lawyer to untangle. Amara had by then a small network of people who could be mobilised: two journalists, a councillor with an interest in urban heat islands, a professor of planning, and about forty former students who could be relied upon to show up. What actually stopped the bulldozers was none of these. It was the survey data. Eleven years of daily records, transcribed and timestamped, constituted the only continuous urban avian dataset for the city, and the environmental impact assessment could not lawfully be signed off without addressing it. The developer's consultants, to their credit, read it properly, and came back with a revised plan that retained the lot as a green corridor because doing so was cheaper than litigating against a decade of evidence.

Baba Sunday did not live to see the corridor formally gazetted. He died in the dry season at eighty-four, in his own bed, and Amara learned of it from a neighbour two days afterward. She went to the garden that evening rather than to the house. The bulbuls were doing what they did at dusk, which was to make an unreasonable amount of noise in the fig, and she stood under it with her notebook and, for the first time in eleven years, did not write anything down.

She took over the exercise books. She had thought this would feel like inheritance and instead it felt like an obligation with a deadline, because she was fifty-three and the whole edifice had twice now rested on a single person's attention. So she did the thing she had been avoiding, which was to make herself unnecessary: she trained four of the neighbourhood's teenagers to run the survey, gave them the protocol in writing, paid them out of a small grant, and deliberately stopped going every day. The data got noisier. It also, for the first time, continued without her.
"""},
    {"domain": "prose", "text": """
The library had been Marcus's refuge since childhood, a cathedral of silence where time moved differently. Now, as the head librarian at sixty-three, he understood that buildings were merely vessels for what people brought to them. The digital age had not killed his beloved institution but transformed it, filling it with laptops and community events, job training and homework help. Some patrons missed the old days of hushed reverence, but Marcus had come to see that libraries had always been about connection, whether through bound pages or internet bandwidth. Today he noticed a young girl, perhaps twelve, who came every day to use the computers. She was preparing college applications, he eventually learned, studying for entrance exams, determined to escape her circumstances. He began setting aside books he thought might help her, leaving them on the table nearest her usual spot with anonymous notes of encouragement. He would never know if she opened them, never receive gratitude, and that was the bargain librarians made with the world: to plant seeds with no guarantee of harvest. His career had held many small victories and immense disappointments, but as he watched the diverse crowd flowing through the doors, heard the murmur of conversations in multiple languages, saw people of all ages finding what they sought, he felt the weight of the work settle comfortably on his shoulders. The library endured because humans needed spaces to grow, to learn, to imagine possibilities beyond their circumstances, and he had dedicated his life to being a guardian of that threshold.

The budget hearing came in March. Marcus had attended eleven of them across his career and had learned that they were not won by eloquence. They were won by numbers arranged so that a council member with four minutes of attention could repeat one of them convincingly to somebody else. So he did not talk about the life of the mind. He talked about the four hundred and twelve tax returns filed from the third-floor computers in the previous year, the two hundred and six people who had used the branch as their mailing address because you cannot apply for work without one, the warming centre days, the number of children in the summer reading programme measured against the summer learning loss figures from the school district. He put the poetry section nowhere in the presentation. It survived because the tax returns did.

Afterward a council aide told him, meaning it kindly, that he was a very effective advocate. He went back to the branch and sat in his office with the door shut for ten minutes, which was as close as he came to anger these days, because the argument that worked was not the argument he believed, and he had made it anyway, and would again.

The girl's name was Deshawn's sister, which is how he first knew her, because Deshawn was a regular and she was the one who came to collect him. Then she started staying. Marcus never asked her name and she never offered it, and this suited them both. He left the books on the near table: a used Princeton Review with someone else's pencil marks erased imperfectly, a battered Baldwin, a guide to financial aid applications that he had ordered specifically and shelved for exactly four days so it would not look ordered specifically. The notes he wrote were short and he threw away three for every one he left. He never saw her open any of them. She took them, which was not the same thing.

She stopped coming in April of that year. This happened constantly and meant nothing in particular; people stopped coming because they moved, because they got work, because something went wrong, and the library was designed around never knowing which. He reshelved the books after a decent interval.

Seven years later a woman came to the desk and asked for him by name. She was in a suit, which is the detail he would remember, and she had to say two sentences before he placed her. She had gone to Rutgers. She was a pharmacist now. She had come back because she was in the city for a conference and had an afternoon and had wanted, she said, to see if the building was still here. She did not mention the books, and Marcus understood that she might not have known, that the notes might have read as the building's ordinary background kindness rather than as anyone's deliberate act, and he decided in the space of about a second that he was not going to tell her.

They talked for twenty minutes about the renovation and the new teen room and her mother's health. Then she left to catch a train.

He did not tell the staff. There was nothing to tell that would survive being said out loud; it would come out sounding like a story with a moral, and it was not that. It was one outcome, unrepresentative, from a practice whose whole premise was that outcomes were not observable. What it gave him was narrower and more useful than vindication: evidence, a single instance, that the mechanism was not imaginary. He went back out to the floor, where a man was having trouble with the printer, and helped him with the printer.
"""},
    {"domain": "prose", "text": """
Elena's bakery occupied the corner storefront her grandmother had owned, where flour dust still seemed to settle in the same way after ninety years, where the morning smell of yeast and butter had seeped into the walls themselves. She had inherited not just a business but a legacy of recipes, techniques, and a clientele spanning generations. The cookies her grandmother had perfected were still made by hand, pressed with the same wooden mold, baked in ovens that hummed a familiar tune. But Elena had also added her own innovations, experimenting with flavors and ingredients, sometimes to the dismay of purists who insisted nothing should change. She had learned to balance preservation with evolution, respecting tradition while not being imprisoned by it. Her teenage son showed interest in the kitchen, asking questions, getting flour in his hair, and she recognized in him the spark that had animated her own youth. She thought about how to pass forward not just recipes but the values they contained: patience, precision, generosity, the understanding that feeding people was a form of love. One evening, preparing dough for the next morning's bake, she found herself humming a tune her grandmother had always sung, a melody that had become part of the rhythm of the kitchen. The music and the memory and the work all blended together, timeless and present simultaneously, and she knew that whatever changes came, this small corner of the world would continue nourishing both body and soul.

The trouble started with the feast of San Rocco, which required four hundred of the almond cookies, the ones pressed in the wooden mold, and which had required four hundred of them every August for longer than Elena had been alive. The mold was a single piece of pearwood, worn concave in the center, and it produced roughly one cookie every nine seconds in the hands of someone who knew what they were doing. Four hundred cookies was an hour of pressing, which was fine, except that the parish had grown and the order that year was eleven hundred.

Her mother would have said no. Elena knew this with certainty, because her mother had said no to the hotel in 1994 and to the wedding company in 2001, on the grounds that the bakery made what it could make properly and that growth which outran the mold was not growth but dilution. Elena had been nineteen during the hotel refusal and had thought it was pride dressed up as principle. She was forty-six now and less sure.

What she did was neither yes nor no. She had a woodworker in Trapani cut three more molds from the same pattern, taking a silicone impression of the original and charging her nearly nine hundred euros, and she hired two of her cousin's daughters for the month of July. The new molds were sharper than the old one. The cookies they produced had crisper edges and a slightly different crumb, because the worn concavity of the original mold packed the dough marginally denser, and Elena discovered this only after the first two hundred had cooled. She stood in the kitchen at eleven at night with a cookie from each mold in either hand, tasting alternately, and understood that she could taste the difference and that perhaps eleven people in the parish could taste the difference and that all eleven of them would tell her.

She sold them anyway, mixed, four hundred from the old mold and seven hundred from the new, and told the parish committee exactly what she had done and why, in writing, before the feast rather than after. Two women complained. One of them, Signora Puglisi, who was eighty-one and had known her grandmother, came into the shop specifically to complain and stayed forty minutes and bought a kilo of the almond paste on the way out.

Nicolò was fifteen that summer and worked the whole of July without once being asked twice. He was fast, faster than Elena had been at his age, and careless in the way fast people are, and she caught herself three separate times about to correct him in her mother's exact cadence and stopping. What she said instead, the third time, was: the mold is worn on the left side, so you have to press half a second longer on the left or the pattern comes out shallow. He said, I know, I figured that out on Tuesday. And then, after a pause, because he was fifteen and could not help himself: you could just get the new one recut deeper on the left and then nobody would have to remember.

Elena thought about it for a long moment, flour to the elbows, the ovens ticking as they cooled.

That's true, she said. We could. Let's not this year, and you can decide when it's yours.

He rolled his eyes, which she had expected, and went back to the tray, which she had also expected. What she had not expected was that he would still be pressing at midnight, and that when she came back from the front he had separated the shallow ones out into a bowl for the family rather than throwing them away, which was precisely what her grandmother had done, and which nobody in that kitchen had ever taught him.
"""},
    {"domain": "prose", "text": """
Thomas had been a war correspondent for thirty years, documenting conflicts across continents, bearing witness to human suffering on scales most people could barely imagine. His photographs had won awards, influenced policy, changed public opinion about distant wars. Yet they haunted him in ways he had never anticipated when young and idealistic, hungry to make a difference through his lens. He had seen things that could not be unseen, understood suffering in textures and dimensions that numbers and headlines failed to convey. When a young photographer asked him for advice, Thomas found himself unable to recommend the path he had chosen. Instead, he spoke about the cost of witnessing, the weight of bearing testimony, the responsibility of accuracy that sometimes felt impossible to carry. He began working with refugees, teaching them to document their own stories, to speak for themselves rather than having narrative imposed upon them by outsiders. It was slower work, less prestigious, less visible to the world, but it felt honest in a way his award-winning career had not. He understood now that distance and objectivity, values taught in journalism schools, could sometimes become armor against empathy, a way to protect yourself from the full weight of what you witnessed. He had learned this late in life, but the learning made the remaining years feel purposeful in a different way, focused less on acclaim and more on genuine service to those whose stories deserved to be told with dignity.

The workshops ran on Tuesdays and Saturdays in a converted shipping office near the port, and the equipment was terrible on purpose. Thomas had started with donated professional bodies and had switched, within four months, to secondhand point-and-shoots and whatever phones people already owned, because the professional cameras produced professional photographs: composed, tonally controlled, and completely evacuated of the thing he was trying to get at. The cheap cameras produced photographs that looked like the person who took them. That was the entire curriculum.

Rasha was nineteen and had been in the country eleven months and photographed, relentlessly, feet. Her own feet, her brother's, the feet of women queuing at the registration office, forty or fifty frames a week of nothing but the lower eighteen inches of the world. Thomas spent the first six weeks resisting the urge to redirect her and the next six understanding why she was right. Faces were dangerous. Faces got you identified, got your family at home identified, got you asked to perform the particular expression of gratitude or suffering that the viewer had come for. Feet were anonymous and specific at once: the shoes told you everything about the journey, the posture told you about the wait, and nobody had ever demanded that a pair of feet look grateful. She had arrived at a documentary strategy that Thomas, with thirty years and two World Press awards, had not thought of, and she had arrived at it out of necessity rather than theory, which is how most real innovations in the form have happened.

The exhibition was at a municipal gallery and Thomas fought hard about two things. The first was captions: the participants wrote their own, in their own languages, with translations placed below rather than replacing them, and nothing was captioned by anyone who had not taken the photograph. The second was the wall text at the entrance, which the gallery's curator had drafted with the phrase giving voice to the voiceless, and which Thomas asked to have removed. The curator was hurt and reasonable about it. Thomas explained, more bluntly than he intended, that they had never been voiceless, that they had been shouting for a decade, and that the wall text was a claim about the gallery's virtue rather than a description of the work. It came down. Something blander went up.

On the second evening of the show he found himself standing in front of one of his own old photographs, which the gallery had included over his objection as context: a 1997 frame from the Balkans, a woman at a roadside, very famous, reproduced in four textbooks. He had taken it from about two meters with a 35mm lens, which meant he had walked up to her, raised a camera to his face, and photographed her at the worst moment of her life without a word passing between them. He had never learned her name. At the time this had seemed like rigor. Standing there at seventy-one with a plastic cup of gallery wine, it seemed like something else, and he found he could hold both facts at once: that the photograph had done measurable good, had moved money and policy, and that he had taken something from her he had never had any right to and could not now return.

Rasha came and stood beside him and looked at it for a while.

Is it a good picture, she asked.

Yes, he said. That's the problem with it.

She thought about that. Then she said that in her language there was a distinction between two words for witness, one meaning the person who saw and one meaning the person who testifies, and that the second one implied being asked. Thomas asked her to write that down for the catalogue. She said she would think about it, and did not, and he understood that this too was her right.
"""},
    {"domain": "prose", "text": """
The retirement home's garden was where Margaret discovered she had not actually died when her doctor said her cancer was terminal. The cancer came and went, in remission for years at a time, making a mockery of certain prognoses. She had become a ghost to many of the people she knew before, awkwardly positioned between life and living, no longer healthy enough for normal existence but too alive to be categorized as dying. In the garden, she found companions in the other residents, fellow travelers in the liminal space between ending and persistence. They played cards, shared books, made jokes about their bodies' betrayals, celebrated small victories like climbing an extra flight of stairs or making it through a meal without pain. She taught them her gardening knowledge, and they taught her about acceptance, about grace in the face of uncertainty. Her daughter visited regularly, and Margaret noticed her daughter had stopped speaking about death with certainty, stopped planning a life as if Margaret would soon no longer be part of it. They developed a new rhythm together, one that didn't negate loss but lived alongside it, that created meaning in the ordinary moments. Margaret had read many books about dying well, but nobody had written about this strange liminal existence, this years-long conversation with mortality. She was living differently now, more present in small moments, less concerned with productivity, more aware of beauty in ordinary things. Time had become both more precious and more flexible, and she was learning, very slowly and with great resistance at first, that uncertainty was not the enemy of peace.

The garden at Fairfield Court was eleven raised beds, a gravel path wide enough for two wheelchairs to pass, and a plastic shed containing forty years of accumulated implements, most of them broken. Margaret's contribution, in her second spring, was to throw out the shed's contents. This was received badly. A resident named Peter Vance, who had run a haulage firm and who spoke to staff in a manner Margaret disliked intensely, objected on the grounds that several of the broken tools were perfectly repairable, and he turned out, infuriatingly, to be correct: he repaired nine of them over the following winter using a vise he had bolted to a bench without permission, and he did it badly and slowly with hands that no longer closed properly, and the tools worked.

They became friends in the specific way of people who have publicly lost an argument to each other. Peter had congestive heart failure and a prognosis that was firmer than Margaret's and shorter. He did not talk about it except in logistics: what he was doing about the firm, which his nephew was ruining, and what he wanted done with the vise. He asked her once, directly, whether she found it easier being uncertain or whether she would rather have a date. She said she had wanted a date for the first two years and now did not. He said he had a date, more or less, and would trade.

The greenhouse was his idea and her execution. He had costed it out on the back of a magazine and it was, characteristically, wrong by a factor of two; she rewrote the proposal, took it to the residents' committee, got it defeated, took it to the family liaison meeting where the relatives of residents were present, and got it funded in nine minutes by three families who wanted very much to be seen to do something. Margaret felt only slightly bad about this. The frame went up in October. Peter supervised from a chair with a blanket over his knees and made continuous observations about the plumb of the uprights, at least two of which were justified.

He died in February, before the glass was finished, which he had known was likely and had arranged for by leaving written instructions about the ventilation that Margaret found folded inside a seed catalogue.

She had expected the grief to be of a particular kind, the kind she had been rehearsing for her own account for six years. It was not. It was ordinary and enormous and had nothing philosophical about it whatsoever. She sat in the half-built greenhouse in the cold for most of an afternoon and her daughter, who had driven up, sat with her and did not say anything useful, which was the correct thing to do.

The tomatoes that first summer were poor. The ventilation instructions were, on inspection, badly wrong, and Margaret left them wrong for the whole season out of a sentimentality she recognised and permitted herself, then fixed them in September.

She was seventy-nine that year. The scan in November was, again, ambiguous; the oncologist used the word stable with the small shrug that had long since stopped alarming her. Walking back across the car park she found herself doing the sum she always did, the one about how many more growing seasons, and noticed that she had stopped doing it as arithmetic and started doing it as a kind of inventory: the greenhouse, the four new beds, the apprentice, a woman of sixty-six named Joan who had arrived in June with no interest in gardening and who was now propagating pelargoniums with an accuracy that bordered on the obsessive. Margaret had not set out to train a successor. She had set out to have company. The one had produced the other without her noticing, which she suspected was the only way it ever worked.
"""},
    {"domain": "prose", "text": """
The city was changing faster than Sergio could document, and he had made it his mission to photograph the disappearing neighborhoods before they were erased and replaced with glass towers and corporate chains. His apartment had become a repository of images, thousands of photographs organized by year and location, documenting the systematic transformation of a place he had grown up in. Each photograph was a small resistance against forgetting, a refusal to let the past be entirely erased by the future. He exhibited his work in community centers and small galleries, showing neighbors their own history, reminding them of streets that had been demolished, buildings that had housed generations. Some people wept looking at his photographs. Others became activated, fighting to preserve what remained. The work was personal and political simultaneously, rooted in love for a place and anger at how places were being treated as commodities to be transformed for profit. His project grew, attracting volunteers who shared his vision, students interested in documentary work, historians who wanted to verify details. What had started as individual passion became a movement, a community effort to document and preserve memory. He understood that photographs couldn't actually save neighborhoods, that the tide of development was larger than any artistic gesture. But they could bear witness. They could insist that what was here mattered, that the lives lived in these spaces had value, that demolition erased more than buildings. And that insistence, that refusal to accept forgetting as inevitable, felt like its own form of resistance.

By the twelfth year the archive had outgrown the apartment. Sergio had converted the second bedroom, then the hall closet, then a rented storage unit on Calle Mendoza that cost him a fifth of his income and that he visited on Sundays with a dehumidifier he did not entirely trust. There were, at the last count he had the stomach to complete, somewhere over ninety thousand negatives and perhaps forty thousand digital frames, organised by street and then by year, in boxes labelled in a handwriting that had deteriorated measurably across the decade.

The problem with an archive is that it is not a body of work. It is raw material that becomes a body of work only when someone imposes an order on it, and Sergio had spent twelve years deferring that act because every order he could imagine was a betrayal of some part of the material. Chronological order made it a story about decline. Geographic order made it a map, which flattened the people out of it. Thematic order, which the curator from the university kept proposing, made it an argument, and he distrusted arguments made out of photographs because photographs will support almost any argument you care to bring to them.

The university wanted the collection. They had wanted it for three years, with increasing formality, and the terms were good: climate-controlled storage, a full-time archivist for two years to catalogue it, digitisation at proper resolution, and open access for researchers. The terms were also, in one respect, impossible. The material would be catalogued according to the institution's descriptive standards, which meant that the neighbourhoods would be indexed by their municipal designations, and the municipal designations were the names assigned by the redevelopment authority in the rezoning of 2009, which were not what anybody had ever called them. La Curva would enter the permanent record as Sector 4B-North.

He argued about this for eleven months. He lost, and then he won, and the winning was smaller than it sounds: the finding aid would carry both names, with the vernacular name first and the municipal designation as a cross-reference, and a scope note explaining the discrepancy. It took four meetings and a letter from a historian at another institution to achieve a semicolon and a footnote. Sergio signed the deed of gift in a conference room in March and went home and could not eat.

What he had not anticipated was how much he would resent the archivist. Her name was Paula, she was twenty-nine, she was extremely good, and within six weeks she had found four hundred frames Sergio had misfiled, identified two buildings he had wrongly labelled, and established, from a shadow and a bus timetable, that a photograph he had dated to 1998 was in fact from 2001. Every correction was a small demonstration that his memory, the thing he had been defending against the developers for twelve years, was itself unreliable. He was rude to her twice. He apologised once, properly, and she accepted it with a matter-of-factness that made it worse.

The exhibition opened in the autumn. He walked through it the night before it opened, alone, as the lighting technician packed up, and found that seeing the work on walls rather than in boxes did something he had not expected: it stopped being his. People he did not know came the following week and stood in front of a photograph of a stairwell on Calle Ocho and told each other things about that stairwell that he had not known and could not have known, and one woman wept in front of a frame he had always considered a technical failure, badly exposed, kept only for the record.

He understood then what the argument about the semicolon had actually been for. Not to preserve the neighbourhood, which was gone, and not to preserve his account of it, which was flawed in ways Paula was still discovering. It was to leave the door open for other people's accounts to be filed alongside his own under a name they would recognise.
"""},
    {"domain": "prose", "text": """
There is a word in the sixth poem that Ilse Brandt has been unable to translate for four months, and she has begun to suspect that the difficulty is not linguistic.

The word is grenzstill. It does not appear in any dictionary she owns, including the 1934 Sprach-Brockhaus that belonged to the poet himself and that she bought at auction in Vienna for an amount she has never disclosed to her sister. It appears exactly once in the surviving corpus, in the penultimate line of a poem about a river in winter, and it is a compound of two entirely ordinary words: border, and still. Grenz. Still. A first-year student could parse it. The trouble is that German compounds do not mean the sum of their parts any more than English ones do — a butterfly is not a fly made of butter — and there is nobody alive to ask what this one meant to the man who made it up.

Ilse works in a rented room above a bakery in Feldkirch, because the poet lived here between 1931 and 1938 and she has a theory, unprovable and probably sentimental, that some words can only be understood in the weather that produced them. The room has a desk, a hot plate, four hundred photocopied pages, and a view of a mountain that is grey for most of the year. In the mornings the bakery's extraction fan makes the window frame hum at a pitch she has stopped hearing. She is sixty-one. This is the fourth poet she has brought into English and she has said, to two separate interviewers, that it will be the last, and she believes this slightly less each time she says it.

The candidate translations fill eleven pages of a notebook. Border-still. Still as a border. Bordered by stillness. Frontier-quiet. The hush at the edge. Each of them is defensible and each of them is wrong in a way she can feel in her sternum before she can articulate it. Border-still is accurate and dead on the page. Frontier-quiet imports an Americanism, a covered wagon, a whole continent of wrong association. The hush at the edge is beautiful and is her own poem rather than his, which is the particular vanity she has spent thirty years training herself to catch.

What she knows: the poem was written in February 1938, five weeks before the Anschluss. The river is the Ill, which does not freeze completely and makes, when partially iced, a sound that Ilse went out in January to hear for herself, standing on the bank for two hours in borrowed boots. The border in question is eleven kilometres away. The poet crossed it in September and did not come back, and the word he coined for whatever he was looking at that February is the last thing he made before the work stops.

Her editor in London, who is kind and who is under no illusions about the commercial prospects of the book, has suggested a footnote. Ilse has considered the footnote seriously for several weeks. A footnote would be honest. It would say: this compound is the poet's invention, it combines the words for border and for stillness, and no English equivalent carries both its ordinariness and its dread. Any reader would understand. The book would be better for it in every measurable respect.

She will not do it, and she has spent some time working out why, and the reason turns out to be this: a footnote is a translator's confession that the poem cannot be brought across, and once you permit yourself one you will permit yourself forty, and what you will have produced is a scholarly edition of a poem rather than the poem. Her whole professional life has been the wager that it can be brought across — not intact, nothing comes across intact, but alive. She is not willing to lose that wager in the second-to-last line.

On the fourteenth of March she walks up the valley road in the afternoon, which she does when the work will not move, and she stops at the place where the road turns and the view opens toward the frontier, and it is completely silent, not peaceful, and she stands there for some time.

That evening she writes stillness at the border in the margin, and crosses it out, and writes below it, in smaller letters, the border-quiet, and looks at it for a long while without deciding anything.

The next morning the bakery fan is louder than usual and she sits down at the desk at ten past six and writes the whole of the sixth poem out fresh in English, all twenty-two lines, without stopping, and when she reaches the penultimate line she puts down a word she had not previously considered and does not much like, and goes on to the end, and leaves it there.
"""},
    {"domain": "factual", "text": """
The process of photosynthesis represents one of nature's most remarkable chemical transformations, converting light energy into chemical energy stored in glucose molecules. In plants, photosynthesis occurs primarily in the chloroplasts of leaf cells, where chlorophyll pigments absorb photons and initiate a cascade of electron transfer reactions. The overall reaction can be summarized as six carbon dioxide molecules plus six water molecules, powered by light energy, producing one glucose molecule and six oxygen molecules. However, this simple equation conceals extraordinary complexity. The process comprises two main stages: the light-dependent reactions occurring in the thylakoid membranes, where photons drive the splitting of water molecules, releasing oxygen as a byproduct and generating adenosine triphosphate and reduced nicotinamide adenine dinucleotide phosphate; and the light-independent reactions or Calvin cycle occurring in the stroma, where these energy carriers drive the fixation of carbon dioxide into organic molecules through a series of enzyme-catalyzed steps. The discovery of photosynthesis's mechanisms required centuries of scientific investigation, from the observations of Joseph Priestley in the 1770s showing that plants somehow restore the air quality of enclosed spaces, through the investigations of Jan Ingenhousz demonstrating light's necessity, to modern molecular techniques revealing the intricate choreography of proteins and electron transfers. Different plants have evolved variations on this basic process: C3 plants using the standard Calvin cycle, C4 plants incorporating an initial fixation step that concentrates carbon dioxide in leaf cells, and CAM plants opening their stomata only at night to conserve water in arid environments. The efficiency of photosynthesis varies dramatically, with theoretical maximum photosynthetic efficiency around ten to thirteen percent of incident solar radiation, though most plants achieve far lower rates due to various limitations including insufficient light capture, imperfect enzyme efficiency, and photorespiration, where Rubisco catalyzes the carboxylation of ribulose-1,5-bisphosphate with oxygen instead of carbon dioxide, reducing productivity.

The chloroplast itself carries the signature of an ancient merger. It possesses its own circular genome, its own ribosomes, and a double membrane, and its genetic sequences place it firmly among the cyanobacteria rather than among plants. The endosymbiotic hypothesis, advanced by Lynn Margulis in 1967 against considerable resistance, holds that a eukaryotic cell engulfed a photosynthetic bacterium roughly one and a half billion years ago and failed to digest it, and that the descendants of that captive cell are the chloroplasts in every leaf. Most of the original bacterial genes have since migrated to the host nucleus, leaving the organelle with perhaps a hundred and twenty of its own, which is why chloroplast assembly requires the coordinated expression of two separate genomes. Red algae and green algae descend from that primary event; several other lineages, including diatoms and dinoflagellates, acquired plastids secondarily by engulfing algae that had already made the bargain, producing organelles wrapped in three or four membranes that record the nesting.

Photorespiration, the process that limits productivity, is best understood as an evolutionary accident that became impossible to reverse. Rubisco evolved when the atmosphere contained very little oxygen, and its active site does not discriminate sharply between carbon dioxide and oxygen because there was no selective pressure to do so. When oxygen levels rose, the enzyme's promiscuity became costly: the oxygenation reaction produces a two-carbon compound that must be salvaged through a pathway spanning the chloroplast, the peroxisome, and the mitochondrion, consuming energy and releasing previously fixed carbon. The cost rises with temperature, because the solubility of carbon dioxide falls faster than that of oxygen as water warms. In a C3 plant on a hot day, photorespiration can consume a quarter or more of the carbon fixed. Rubisco is also remarkably slow, turning over perhaps three molecules per second where a typical enzyme manages thousands, which is why it must be present in enormous quantity and why it is, by mass, the most abundant protein on the planet.

The C4 pathway is a biochemical workaround that has evolved independently more than sixty times, a striking case of convergence. Carbon dioxide is first fixed in mesophyll cells by PEP carboxylase, an enzyme with no affinity for oxygen at all, producing a four-carbon acid that is shuttled into bundle sheath cells and decarboxylated there. This delivers carbon dioxide to Rubisco at concentrations high enough to suppress the oxygenation reaction almost entirely. The arrangement costs additional energy per carbon fixed but pays for itself in hot, bright, or dry conditions, which is why maize, sugarcane, and sorghum are C4 and why they outproduce wheat and rice per unit of water. CAM plants, including cacti and pineapple, separate the same two steps in time rather than space, opening their stomata only at night to capture carbon dioxide into organic acids and processing it behind closed stomata during the day, an adaptation that reduces water loss by an order of magnitude at the cost of slower growth.

The global consequences are measured in tens of billions of tonnes. Terrestrial and marine photosynthesis together fix roughly one hundred and twenty gigatonnes of carbon annually as gross primary production, about half of which is immediately returned by plant respiration. Marine phytoplankton, despite representing well under one percent of photosynthetic biomass, account for nearly half of global primary production because their turnover is measured in days rather than decades. This flux dwarfs annual anthropogenic emissions of roughly ten gigatonnes, which is precisely why relatively small imbalances between photosynthesis and respiration dominate the carbon cycle's response to warming.

Efforts to engineer improvements are among the most ambitious projects in plant science. The RIPE consortium demonstrated yield increases near twenty percent in field trials of tobacco by accelerating the recovery of photosystem II from its photoprotective state, which normally relaxes over minutes and wastes carbon fixation whenever a leaf passes from sun into shade. Other groups have transplanted bacterial carbon-concentrating mechanisms into crop chloroplasts, or attempted the far harder task of installing the complete C4 anatomy into rice, a project running since 2008 that must coordinate changes in leaf venation, cell wall properties, and the expression of a dozen enzymes. Synthetic carbon fixation cycles designed in vitro have achieved rates exceeding Rubisco's, though installing them in a living organism without disrupting central metabolism remains unsolved.
"""},
    {"domain": "factual", "text": """
The Great Wall of China represents one of human civilization's most ambitious engineering projects, stretching approximately thirteen thousand miles across northern China in a series of fortifications built and rebuilt over more than two millennia. The earliest walls were constructed beginning in the seventh century BCE by various Chinese states seeking to defend against incursions from nomadic groups in the north, but the structure people recognize today was largely built during the Ming Dynasty between the fourteenth and seventeenth centuries CE. The wall's construction involved millions of workers over centuries, including soldiers, peasants, and prisoners, with estimates suggesting that significant portions required between two hundred and three hundred years of continuous labor. The wall typically consists of brick or stone facing over a rubble core, with a walkway on top wide enough for horses, beacon towers at regular intervals, and garrison stations spaced along its length. The most visited and well-preserved sections near Beijing demonstrate the technical sophistication of Ming-era construction, featuring precisely cut stones fitted without mortar, sloping profiles that shed water and resist erosion, and crenellations designed for defensive advantage. Archaeological evidence indicates that the wall's effectiveness as a defensive barrier was limited; instead, it served primarily as a border control mechanism, regulating trade along the Silk Road and collecting taxes on goods passing through designated passes. The wall also played crucial roles in military signaling, with beacon fires communicating messages across vast distances, and in psychological assertion of imperial power, representing a concrete manifestation of the state's ability to mobilize resources and control territory. The construction of the wall exacted enormous costs in human suffering, with millions dying during its building, yet it became culturally central to Chinese identity, symbolizing perseverance, sacrifice, and the greatness of Chinese civilization. Modern preservation efforts have focused on stabilizing remaining sections while balancing the demands of tourism, scientific study, and authentic historical conservation.

The construction technique that dominates by length is not masonry but rammed earth, known as hangtu. Workers erected parallel wooden formwork, filled it with a few inches of soil mixed with lime and sometimes gravel, and compacted it with weighted rams until it rang; the formwork was then raised and the process repeated. A well-built rammed earth wall achieves compressive strength comparable to weak concrete and can survive two thousand years in a dry climate, which is why substantial stretches of Han dynasty wall still stand in the Gansu corridor while far younger brick sections have collapsed. In the arid west, where soil was poor, builders layered reeds and tamarisk branches between earth courses, and these organic layers have proved invaluable to archaeologists because they can be radiocarbon dated directly.

The first unification of the walls occurred under Qin Shi Huang after 221 BCE, when the newly consolidated empire linked existing state walls into a continuous northern frontier under the general Meng Tian. The human cost of this campaign entered Chinese literature permanently through the legend of Meng Jiangnü, whose weeping for her conscripted husband was said to have collapsed a section of the wall and revealed the bones of the dead within it. The Qin walls lay considerably north of the Ming construction that tourists visit, and almost nothing of them survives above ground.

Logistics governed everything. Feeding the garrison was a greater problem than building the wall, because transporting grain overland consumed a large fraction of what was carried; the Han addressed this with tuntian, military-agricultural colonies in which soldiers farmed the land they defended. Ming records document brick kilns operating near the wall line with stamped bricks bearing the kiln, the date, and the supervising officer's name, an accountability system that allowed defective work to be traced to its maker. Mortar in the best Ming sections incorporated sticky rice, whose amylopectin produced a composite of remarkable durability and resistance to water penetration, a fact confirmed by analysis only in 2010.

The wall's defensive record is equivocal. It did not stop the Mongols: Genghis Khan's forces passed it in the early thirteenth century, and Kublai Khan ruled all of China from 1279. It did not stop the Manchus in 1644, who entered through the Shanhai Pass when the commanding general Wu Sangui opened the gate to them during a dynastic collapse. What the wall did reliably was raise the cost of raiding, since mounted parties could cross but could not easily return across a garrisoned line while driving captured livestock, and the beacon system could mobilize a response faster than raiders could withdraw. Signals used smoke by day and fire by night, with the number of columns encoding the size of the approaching force, and messages could traverse hundreds of kilometres within hours.

Two persistent myths deserve correction. The wall is not visible from the Moon, nor from low Earth orbit under normal conditions; it is roughly the width of a highway and runs across terrain of similar colour, and astronauts have repeatedly confirmed they cannot pick it out unaided. Nor is it a single continuous structure. Comprehensive survey work by China's State Administration of Cultural Heritage, completed in 2012 using GPS and infrared imaging, placed the total length of all walls, trenches, and natural barriers across all dynasties at 21,196 kilometres, comprising more than ten thousand discrete segments, many of them parallel or redundant lines built centuries apart.

Conservation faces pressures that are demographic as much as environmental. Roughly thirty percent of the Ming wall has disappeared entirely and a further large fraction is classified as in poor condition, with losses driven by wind erosion in the west, vegetation and freeze-thaw damage in the east, and, historically, villagers quarrying dressed stone and brick for houses and livestock enclosures. A 2006 regulation made removing bricks a punishable offence. The most heavily restored sections near Beijing, particularly Badaling, receive tens of thousands of visitors daily and have been rebuilt so thoroughly that conservators debate whether they constitute restoration or reconstruction, while the so-called wild wall favoured by hikers degrades under foot traffic with no supervision at all.
"""},
    {"domain": "factual", "text": """
The development of written language represents a watershed moment in human cognitive and social evolution, fundamentally altering how knowledge could be preserved, transmitted, and built upon across generations. The earliest writing systems emerged independently in multiple regions between approximately 3400 and 1200 BCE, including cuneiform in Mesopotamia, hieroglyphics in Egypt, and logographic systems in China. Cuneiform, developing from earlier token systems used for accounting, represents humanity's oldest known writing system, initially recording economic transactions using wedge-shaped impressions on clay tablets before expanding to include literary, legal, and religious texts. The Sumerian scribes who developed cuneiform possessed no concept that they were inventing writing; rather, they were adapting existing token-based accounting systems into a visual notation that could preserve quantitative information. Over centuries, cuneiform expanded to represent phonetic elements, eventually enabling the recording of spoken language itself. Egyptian hieroglyphic writing developed similarly, combining logographic symbols with phonetic elements, and persisting for over three thousand years with remarkable consistency. Unlike cuneiform's medium of clay, Egyptian scribes used papyrus, a material more fragile but easier to write upon, enabling larger-scale textual production. Chinese writing systems, developing from oracle bone inscriptions, maintained stronger continuity with their logographic roots, with individual characters representing both sound and meaning, a characteristic distinguishing Chinese from most other writing systems. The development of alphabetic systems, where individual symbols represent phonetic elements rather than words, occurred later but enabled more efficient literacy, since learning a smaller set of characters granted access to any word in a language. The Phoenicians, Hebrews, and Greeks all developed early alphabetic systems, with the Greek alphabet eventually influencing the Latin alphabet that dominates contemporary global communication. The transition from orality to literacy fundamentally altered human cognition, with reading and writing engaging different neural pathways than spoken language, while simultaneously enabling the accumulation of knowledge across centuries and the development of complex administrative, legal, and philosophical systems that defined civilization.

Recovering these systems after their scripts fell out of use has been among the great intellectual achievements of the modern era, and the methods differ sharply depending on what the decipherer has to work with. Egyptian hieroglyphs were closed to readers for some fourteen centuries until the Rosetta Stone, recovered in 1799, supplied the same decree in hieroglyphic, demotic, and Greek. Even with a bilingual text the problem was not trivial. Thomas Young established that the cartouches enclosed royal names and that some signs were phonetic, but it was Jean-François Champollion, who read Coptic fluently and recognised it as the descendant of the ancient language, who in 1822 demonstrated that the script was neither purely pictographic nor purely alphabetic but a mixed system combining phonetic signs, logograms, and unpronounced determinatives that disambiguated meaning.

Linear B presented the opposite situation: no bilingual text existed, and the underlying language was unknown. Alice Kober, working through the 1940s on index cards cut from the backs of greeting cards during wartime paper shortages, identified sets of words that differed only in their final signs, proving the language was inflected and establishing which signs shared consonants or vowels without knowing the value of any. Michael Ventris built on her grids and announced in 1952 the conclusion he had least expected, having assumed the language was Etruscan: the tablets were written in an archaic form of Greek, five centuries older than Homer. The texts turned out to be inventories almost exclusively — chariot wheels, sheep, rations, offerings to deities — which disappointed those hoping for literature but confirmed that writing in the Aegean began, as it had in Mesopotamia, with accountancy.

Maya glyphs resisted longer because of a scholarly error. Eric Thompson, who dominated the field for decades, insisted the script was purely ideographic and non-linguistic, and his authority suppressed the phonetic approach until Yuri Knorozov, working in Moscow with poor reproductions and no access to the sites, showed in 1952 that the signs included a syllabary. Full decipherment followed over the subsequent forty years, and the result transformed Maya studies from the archaeology of an allegedly peaceful astronomical civilisation into the political history of competing dynasties with named kings, recorded wars, and dated accessions.

Some systems remain unread. The Indus Valley script, attested on several thousand short seals averaging fewer than five signs, may not encode language at all; the Proto-Elamite tablets and Linear A both await either a bilingual text or a recognised underlying language.

The mechanisation of writing altered its social distribution more than its structure. Woodblock printing in Tang China preceded Gutenberg by centuries, and Bi Sheng produced movable ceramic type around 1040, but the enormous character inventory of Chinese limited its advantage, whereas an alphabet of a few dozen sorts made European movable type overwhelmingly efficient. Within fifty years of Gutenberg's Bible, European presses had produced perhaps twenty million volumes, collapsing the price of a book by more than an order of magnitude and making the Reformation's pamphlet war possible. Literacy rates, which had stood in the low single digits across most of medieval Europe, rose slowly and then sharply, reaching near-universality in northwestern Europe only in the nineteenth century with compulsory schooling.

Digital encoding posed the same problem of representation in a new form. Early systems such as ASCII allocated seven bits and encoded only the Latin alphabet, forcing every other writing system into incompatible national encodings that garbled text when exchanged. Unicode, begun in 1987, assigns every character in every known script a unique code point, currently defining more than 149,000 of them across 161 scripts, including several that have no living readers. The technical difficulties recapitulate the linguistic ones: scripts that run right to left, scripts in which letters change shape according to their neighbours, and scripts such as Devanagari in which consonant clusters fuse into ligatures all require rendering rules that go well beyond mapping numbers to shapes.
"""},
    {"domain": "factual", "text": """
The immune system represents an extraordinarily sophisticated biological defense network evolved over hundreds of millions of years to protect organisms from pathogenic microorganisms and aberrant cells. Immunology as a scientific discipline emerged in the late nineteenth and early twentieth centuries, beginning with Louis Pasteur's studies of vaccination and progressing through the twentieth century to contemporary molecular understanding of immune mechanisms. The immune system comprises numerous distinct but integrated components: physical barriers like skin and mucous membranes providing the first line of defense, innate immunity mechanisms responding within hours to perceived threats through pattern recognition and inflammation, and adaptive immunity involving lymphocytes that generate specific responses to particular pathogens. The innate immune system activates through pattern recognition receptors that detect molecular structures shared by groups of pathogens rather than responding to individual pathogens, triggering inflammatory responses that contain infection and destroy invaders. Complement proteins in blood plasma form cascades of reactions amplifying inflammatory responses and directly destroying some pathogens. Phagocytic cells including neutrophils and macrophages engulf and destroy bacteria and damaged cells, while natural killer cells identify and destroy cells displaying stress signals or viral proteins. The adaptive immune system develops specific responses through B cells producing antibodies and T cells providing cellular immunity, with this system requiring days or weeks to mount responses but generating immunological memory enabling rapid responses upon reencounter with specific pathogens. This memory forms the basis of vaccination, where exposure to harmless forms of pathogens generates immune memory without severe disease. Dysregulation of immune systems leads to diseases ranging from immunodeficiency syndromes like HIV to autoimmune conditions where the immune system attacks self-tissues, while overactive inflammatory responses cause tissue damage and systemic dysfunction. Contemporary immunology involves increasingly sophisticated understanding of cytokine signaling, T-regulatory cell development, checkpoint mechanisms that prevent excessive immune responses, and therapeutic interventions ranging from monoclonal antibodies to engineered T cells, representing some of the most dynamic areas of modern biomedical science.

The central puzzle of adaptive immunity is how a genome of some twenty thousand genes produces receptors capable of binding essentially any molecular shape, including synthetic compounds no organism has ever encountered. The answer, established by Susumu Tonegawa in 1976 and recognised with a Nobel Prize eleven years later, is that lymphocytes rearrange their own DNA. The receptor gene exists in the germline as separate libraries of variable, diversity, and joining segments; during lymphocyte development, the RAG1 and RAG2 enzymes excise the intervening DNA and splice one segment from each library together, with additional nucleotides inserted or deleted at the junctions by terminal deoxynucleotidyl transferase. Combinatorial choice among segments, junctional imprecision, and the pairing of two independently rearranged chains together generate a theoretical repertoire exceeding ten to the eleventh distinct receptors. B cells then refine their receptors further through somatic hypermutation in germinal centres, introducing point mutations at a rate a million times the background and selecting the variants that bind antigen most tightly, a process of directed evolution running on a timescale of days.

This generative power creates an obvious hazard: receptors are produced at random and many will recognise the body's own tissues. Tolerance is enforced at two stages. In the thymus, developing T cells are presented with self-peptides displayed on MHC molecules, and those binding too weakly to be useful die by neglect while those binding too strongly are deleted, a double filter termed positive and negative selection that eliminates the great majority of thymocytes. The AIRE transcription factor permits thymic epithelial cells to express genes normally restricted to peripheral organs — insulin, thyroid proteins — so that T cells can be screened against tissues they will never otherwise encounter during development; mutations in AIRE produce a syndrome of multiple simultaneous autoimmune diseases. Peripheral tolerance supplies a second layer through regulatory T cells, which suppress responses to self-antigens and whose absence, in mutations of the FOXP3 gene, causes fatal multi-organ autoimmunity in infancy.

MHC molecules, called HLA in humans, are the most polymorphic genes known, with thousands of alleles at some loci. This diversity is maintained by balancing selection: a population in which everyone displayed peptides identically would be vulnerable to any pathogen that evolved to avoid that presentation. The same polymorphism makes transplantation difficult, since foreign MHC provokes exceptionally strong rejection, and it explains why particular HLA types confer susceptibility to specific autoimmune conditions, most famously HLA-B27 in ankylosing spondylitis.

Much of the immune system's work occurs at mucosal surfaces, which present a surface area far larger than the skin and which are colonised by trillions of commensal organisms that must be tolerated rather than attacked. Secretory immunoglobulin A, produced in greater quantity than all other antibody classes combined, coats these surfaces and restrains bacteria without triggering inflammation. The gut microbiota actively shapes immune development: germ-free mice display stunted lymphoid tissue and skewed T cell populations, and particular commensal species induce regulatory T cells that dampen systemic inflammation. Disruption of this relationship is implicated in inflammatory bowel disease and in the rising incidence of allergy in industrialised populations.

Allergy itself is a misdirected response in which IgE antibodies, evolved to combat parasitic worms, are generated against harmless proteins such as pollen or peanut allergens. IgE binds receptors on mast cells, and subsequent exposure cross-links those receptors, triggering degranulation and the release of histamine and other mediators within seconds, producing symptoms ranging from rhinitis to fatal anaphylaxis.

Therapeutic manipulation has advanced rapidly. Checkpoint inhibitors block the PD-1 and CTLA-4 pathways that tumours exploit to switch off infiltrating T cells, producing durable remissions in melanoma and lung cancer that were previously unattainable. CAR-T therapy removes a patient's own T cells, transduces them with a synthetic receptor recognising a tumour surface protein such as CD19, expands them, and returns them, achieving complete response rates above eighty percent in refractory B cell leukaemias while risking cytokine release syndrome severe enough to require intensive care.
"""},
    {"domain": "factual", "text": """
The transition from hunter-gatherer societies to agricultural civilization, occurring at different times in different regions between approximately twelve thousand and five thousand years ago, fundamentally transformed human societies in ways whose consequences continue shaping our present world. The Neolithic Revolution, as this transition is termed, involved the deliberate cultivation of plants and the domestication of animals, requiring sustained settlement, long-term planning, and technological innovations including irrigation, plow agriculture, and food storage systems. The transition occurred first in the Fertile Crescent around ten thousand years ago, in China approximately nine thousand years ago, in Mesoamerica around eight thousand years ago, and independently in other regions including Sub-Saharan Africa, the Andes, and New Guinea. Agricultural societies enabled population growth by producing more calories per unit area than hunting-gathering, requiring approximately ten times more land per person, but allowing those lands to support denser populations through stored grain and root crops. Agricultural surplus enabled specialization of labor, allowing some individuals to focus on crafts, administration, warfare, or religion rather than food production. This specialization facilitated the emergence of complex societies with hierarchical organization, formalized governance systems, and monumental architecture reflecting concentrated labor and resources. The shift to agriculture correlated with increasing human-animal contact, creating opportunities for zoonotic disease emergence, with pathogens including measles, influenza, and smallpox jumping from domesticated animals to human populations. Agricultural societies developed writing systems to manage increased administrative complexity and trade, though literacy remained restricted to small populations in many societies. The social stratification accompanying agriculture created classes of rulers, priests, merchants, and enslaved peoples, distinct from the more egalitarian social structures of most hunter-gatherer societies. Agriculture enabled civilization's great achievements in art, philosophy, technology, and science, while simultaneously enabling organized warfare and genocide. Contemporary debates persist regarding whether the agricultural transition represented progress or created the foundation for human suffering and environmental degradation.

The skeletal evidence weighs heavily on the pessimistic side of that debate. Populations that adopted cereal agriculture show a consistent suite of changes: adult stature declines, in some Eastern Mediterranean sequences by more than ten centimetres between the Palaeolithic and the early Neolithic; dental caries increase sharply with the shift to starch-rich diets; enamel hypoplasia, which records arrested growth during childhood illness or hunger, becomes markedly more common; and porotic hyperostosis, associated with iron-deficiency anaemia, appears in populations where it had been rare. Life expectancy did not improve and in several regions fell. What rose was fertility, because sedentism shortened birth intervals — a mobile forager must carry a child until it can walk, constraining births to roughly four-year gaps, whereas a settled population with weaning foods of cereal gruel need not. Agriculture therefore spread not because it made individuals healthier but because it made populations grow faster, and larger populations displaced smaller ones.

The domestication process itself is legible in the genetics and the morphology of the crops. Wild wheats and barleys shatter at maturity, dispersing seed via a brittle rachis; the domesticated forms carry mutations producing a tough rachis that retains grain on the stalk, a trait lethal in the wild but strongly favoured by harvesting with sickles, since only the seed still attached is gathered and resown. Archaeobotanical sequences from sites such as Abu Hureyra and Çatalhöyük show the proportion of tough-rachis grains rising gradually across one to two thousand years, indicating that domestication was a slow unconscious process rather than a discovery. Comparable changes mark animal domestication: reduced body size, shortened faces, smaller brains, and retained juvenile behaviours, the constellation sometimes called domestication syndrome and hypothesised to follow from selection on neural crest development.

Stable isotope analysis of bone collagen has made dietary reconstruction far more precise. Carbon isotope ratios distinguish C3 from C4 plants, allowing maize consumption to be tracked through the Americas, while nitrogen ratios indicate trophic level and can separate marine from terrestrial protein. Such work has overturned tidy replacement narratives. In coastal Europe the transition to farming was in places abrupt and in places prolonged over centuries of mixed subsistence, and ancient DNA has shown that the Neolithic arrived in Europe primarily through the migration of Anatolian farming populations who then admixed with, rather than simply replaced, resident hunter-gatherers.

Lactase persistence supplies the clearest instance of agriculture reshaping human biology. Mammals normally cease producing lactase after weaning; mutations in the regulatory region upstream of the LCT gene maintain production into adulthood, and these arose independently at least four times — once in Europe, and separately in East African and Arabian pastoralist populations. The European variant shows one of the strongest signals of recent positive selection in the human genome, sweeping to high frequency within a few thousand years, and notably it postdates dairying by a considerable interval, indicating that people processed milk into low-lactose cheese and yoghurt long before they could drink it fresh.

The secondary products revolution, a term introduced by Andrew Sherratt, describes a second transformation some three to four millennia after initial domestication, in which animals began to be exploited for renewable outputs rather than meat alone: milk, wool, traction, and transport. Ploughing with oxen permitted cultivation of heavy soils beyond the reach of hoe agriculture and multiplied the area a household could work; wool made textile production a storable, tradeable industry; and pack animals extended exchange networks. Each of these amplified inequality, since traction animals and flocks are heritable capital in a way that foraging skill is not, and archaeological burials from this period show wealth differentials of a magnitude absent from earlier cemeteries.
"""},
    {"domain": "factual", "text": """
The development of quantum mechanics in the early twentieth century revolutionized physics, replacing classical Newtonian mechanics as the fundamental framework for understanding atomic and subatomic phenomena. Classical physics, developed primarily by Isaac Newton and elaborated by subsequent researchers, successfully described macroscopic phenomena and remained adequate for engineering and most practical applications for more than two centuries. However, observations at the atomic scale revealed phenomena inexplicable within classical frameworks, including atomic absorption and emission of specific discrete frequencies of light, the stability of atoms despite electrons circling positively charged nuclei, and effects where particles demonstrated properties of waves. Planck's proposal that energy came in discrete quanta, Einstein's photoelectric effect explanation, and Bohr's model of atomic structure with discrete energy levels began reconciling these observations, but the full framework emerged through the work of Heisenberg, Schrödinger, Dirac, and others in the mid-1920s. Schrödinger's wave equation describes how the quantum state of a system evolves, with the wave function containing all available information about a quantum system, though interpretation of what the wave function represents remains contested. Heisenberg's uncertainty principle establishes fundamental limits on simultaneous knowledge of certain pairs of properties like position and momentum, not reflecting limitations of measurement but rather fundamental properties of nature at quantum scales. Quantum mechanics introduces inherent probabilistic elements, with measurement outcomes uncertain until observed, contradicting classical determinism and generating philosophical debates about the nature of reality and observation. The Pauli exclusion principle explains the shell structure of atoms and the periodic table, while quantum tunneling explains phenomena from radioactive decay to electron microscopy. Contemporary applications of quantum mechanics include semiconductors and lasers, nuclear energy, magnetic resonance imaging, and emerging quantum computing. Quantum mechanics successfully integrates with special relativity in quantum field theory, the framework underlying particle physics and explaining fundamental forces.

The sharpest conceptual difficulty concerns entanglement. When two particles interact and separate, the pair may occupy a joint state that cannot be written as a product of individual states, so that neither particle possesses a definite property of its own while the pair possesses a definite correlation. Einstein, Podolsky and Rosen argued in 1935 that this demonstrated the theory's incompleteness: measuring one particle appears to fix the other instantaneously at arbitrary distance, which they took as evidence that hidden variables must have carried the outcome all along. The argument remained philosophical until John Bell showed in 1964 that it was experimentally decidable. Bell derived an inequality that any theory respecting local realism must satisfy, and quantum mechanics predicts violations of it. Experiments by Alain Aspect in 1982, and progressively more rigorous versions closing the detection and locality loopholes simultaneously in 2015, have confirmed the violations decisively; the 2022 Nobel Prize recognised this work. Nature is not locally realistic. Crucially, entanglement transmits no usable signal, because the local outcomes are random and the correlation becomes visible only when the two records are compared through a classical channel, so relativistic causality survives intact.

What measurement actually is remains contested. The Copenhagen interpretation treats collapse as a primitive, declining to model the apparatus quantum mechanically. The many-worlds interpretation removes collapse entirely, taking unitary evolution as complete and accepting that the observer becomes entangled with the outcomes, all of which occur in decohered branches. Pilot-wave theory restores determinism by positing definite particle positions guided by the wave function, at the cost of explicit non-locality. Objective collapse models add stochastic terms to the Schrödinger equation and are, unlike the others, experimentally distinguishable in principle. Decoherence theory, developed from the 1970s, explains why superpositions are unobservable at macroscopic scale without settling the interpretive question: coupling to environmental degrees of freedom destroys phase coherence on timescales that fall extraordinarily fast with system size, so that a dust grain in a vacuum illuminated only by the cosmic microwave background loses coherence in far less than a microsecond.

Quantitatively, quantum electrodynamics is the most precisely tested theory in science. The electron's anomalous magnetic moment has been calculated to tenth order in perturbation theory and measured in a Penning trap, with theory and experiment agreeing to roughly twelve significant figures, an accuracy equivalent to measuring the distance from New York to Los Angeles to within the width of a human hair. The same framework, extended to the strong and weak interactions, constitutes the Standard Model, whose final predicted particle, the Higgs boson, was observed at CERN in 2012.

Collective quantum effects produce some of the most striking macroscopic phenomena. Superconductivity arises when electrons form Cooper pairs through a phonon-mediated attraction, and because pairs are bosons they condense into a single coherent state with zero electrical resistance, expelling magnetic fields entirely. Superfluid helium climbs container walls and flows without viscosity. Both are quantum mechanics made visible at human scale.

Quantum computing exploits superposition and entanglement to encode information in ways classical machines cannot efficiently simulate. A register of n qubits occupies a state described by two to the n complex amplitudes, and algorithms such as Shor's factoring procedure achieve exponential speedup over the best known classical methods by arranging interference so that wrong answers cancel. The practical obstacle is decoherence: physical qubits lose coherence within microseconds to milliseconds, so error correction schemes encode each logical qubit across many physical ones, with current estimates requiring roughly a thousand physical qubits per logical qubit for cryptographically relevant computations. Devices demonstrated so far operate in the noisy intermediate-scale regime, with hundreds of physical qubits and no full error correction, and the threshold at which quantum machines outperform classical ones on commercially meaningful problems remains unreached.
"""},
    {"domain": "factual", "text": """
Plate tectonics is the unifying theory of the solid Earth, holding that the planet's rigid outer shell is fragmented into about fifteen major plates that move relative to one another at rates of a few centimetres per year, comparable to the growth of a fingernail. The theory's acceptance in the 1960s constituted a genuine scientific revolution, resolving a century of disconnected observations into a single mechanism.

Alfred Wegener assembled the case for continental drift in 1912, marshalling the jigsaw fit of the Atlantic coastlines, matching Permian glacial deposits across South America, Africa, India and Australia, and identical fossils of the reptile Mesosaurus on both sides of an ocean it could not have crossed. His evidence was strong and his mechanism was not: he proposed that continents ploughed through oceanic crust, driven by tidal and centrifugal forces, and physicists correctly demonstrated that the forces were orders of magnitude too weak and the rock far too strong. The hypothesis was rejected for four decades largely on that basis.

The evidence that revived it came from the sea floor, mapped in detail only after wartime development of sonar and magnetometry. Surveys revealed a globe-encircling mid-ocean ridge system, sixty thousand kilometres long, with a rift valley along its crest. Harry Hess proposed in 1962 that new oceanic crust was created at these ridges and carried outward, a process Robert Dietz named sea-floor spreading. The decisive confirmation came from magnetism. Basalt erupting at a ridge cools through the Curie temperature and locks in the direction of the ambient magnetic field, and the Earth's field reverses polarity at irregular intervals averaging a few hundred thousand years. Fred Vine and Drummond Matthews predicted, and surveys confirmed, symmetrical stripes of alternating magnetic polarity running parallel to ridges and mirrored across them. The sea floor is a tape recording of the geomagnetic field, and it is young everywhere: no oceanic crust older than about two hundred million years exists, against continental rocks approaching four billion.

Three plate boundary types account for most geological activity. At divergent boundaries plates separate and mantle material rises to fill the gap, decompression melting producing basaltic crust; Iceland exposes this process above sea level, and the East African Rift shows a continent in the early stages of splitting. At convergent boundaries plates collide, and the outcome depends on what is colliding. Oceanic lithosphere, which grows denser as it cools, subducts beneath either oceanic or continental plates, producing deep trenches, volcanic arcs, and earthquakes along an inclined Wadati-Benioff zone traceable to seven hundred kilometres depth. Water driven off the descending slab lowers the melting point of the overlying mantle wedge, generating the explosive, silica-rich magmas of the Pacific Ring of Fire. Where two continents meet, neither subducts readily because continental crust is too buoyant, and the collision thickens the crust instead: the Himalaya and the Tibetan Plateau record India's ongoing impact with Asia, which began roughly fifty million years ago and continues at about five centimetres per year. At transform boundaries plates slide past one another, as along the San Andreas Fault, producing earthquakes without volcanism.

The driving mechanism is now understood to reside mainly in the plates themselves rather than in mantle convection dragging them along. Slab pull, the gravitational sinking of dense subducted lithosphere, supplies the dominant force, with ridge push from the elevated topography of spreading centres contributing less. Plates with long subducting margins, such as the Pacific, move several times faster than plates with none.

Earthquake magnitude is reported on the moment magnitude scale, which measures the seismic moment: the product of fault area, average slip, and rock rigidity. The scale is logarithmic, so each unit represents a thirty-two-fold increase in energy released. The 1960 Valdivia earthquake in Chile, magnitude 9.5, remains the largest instrumentally recorded, rupturing roughly a thousand kilometres of the subduction interface. Prediction of individual earthquakes has proved intractable and is regarded by most seismologists as unattainable; forecasting, which assigns probabilities over decades and informs building codes and land use, has proved both achievable and effective, and the disparity in death tolls between earthquakes of similar magnitude in countries with and without enforced seismic codes is the clearest demonstration of its value.

Hotspot volcanism supplies one of the theory's most elegant confirmations and also one of its unresolved problems. The Hawaiian islands form a chain whose ages increase monotonically to the northwest, from active eruption on the Big Island to extinct, eroded seamounts thousands of kilometres away, and the chain bends sharply at the Emperor seamounts around forty-seven million years ago. The standard explanation, proposed by J. Tuzo Wilson in 1963, is a stationary plume of hot mantle material rising from great depth, with the Pacific plate sliding over it like paper drawn across a candle flame. Whether such plumes genuinely originate at the core-mantle boundary, and whether the Emperor bend records a change in plate motion or a migration of the plume itself, remain actively contested, and seismic tomography has so far imaged deep plume structures beneath only a minority of proposed hotspots.

The supercontinent cycle operates on a timescale of roughly four hundred to six hundred million years. Pangaea, assembled by about three hundred and thirty-five million years ago and beginning to break apart around one hundred and seventy-five million years ago, is merely the most recent; Rodinia preceded it by some seven hundred million years, and Columbia before that. Reconstructing these earlier configurations relies on palaeomagnetism, since the inclination of remanent magnetisation in a rock records the latitude at which it formed, though longitude remains unconstrained by the method. The assembly and dispersal of supercontinents exerts first-order control on climate, sea level, and evolution: continental interiors far from oceanic moisture become arid, shallow shelf seas expand and contract with rifting, and the isolation or reconnection of biotas drives speciation and extinction.

Plate tectonics also appears to be rare. Neither Venus, Mars, nor Mercury exhibits it despite Venus being nearly Earth's twin in size and composition. The prevailing explanation invokes water: Earth's mantle contains water that weakens olivine and permits the lithosphere to bend and subduct, whereas Venus, having lost its water to atmospheric escape after a runaway greenhouse, possesses a lithosphere too strong and buoyant to founder, and instead appears to resurface catastrophically at intervals of several hundred million years. If correct, this links plate tectonics to the long-term habitability of a planet, since subduction drives the carbonate-silicate cycle that regulates atmospheric carbon dioxide over geological time, acting as a thermostat that has kept liquid water stable on Earth's surface for billions of years despite a gradually brightening Sun.
"""},
    {"domain": "factual", "text": """
The Global Positioning System is an engineering achievement that is also, incidentally, the most widely deployed practical application of Einstein's theories of relativity. Conceived by the United States Department of Defense in the early 1970s and reaching full operational capability in 1995, it provides position and time anywhere on Earth, continuously and without charge, to an unlimited number of users.

The constellation comprises at least twenty-four operational satellites in six orbital planes inclined at fifty-five degrees, at an altitude of roughly 20,200 kilometres, each completing two orbits per sidereal day. This geometry guarantees that at least four satellites are visible from any point on the surface at any time, a requirement that follows directly from the mathematics of the fix.

The principle is trilateration by timing. Each satellite broadcasts a signal encoding the exact moment of transmission and the satellite's own orbital parameters. A receiver measures the elapsed travel time and multiplies by the speed of light to obtain a range. Three such ranges would suffice to determine a position in three dimensions if the receiver's clock were perfect. It is not: consumer receivers carry quartz oscillators, and an error of one microsecond corresponds to a position error of three hundred metres. The elegant solution is to treat the clock offset as a fourth unknown and solve for it simultaneously, which requires a fourth satellite. The consequence is that every GPS receiver is also a precision clock, disciplined to the atomic standards aboard the satellites, and this timing function has become at least as economically important as the positioning function: telecommunications networks, electrical grid synchronisation, and financial transaction timestamping all depend on it.

Relativity enters unavoidably. Special relativity predicts that the satellite clocks, moving at about 3.9 kilometres per second, run slow relative to ground clocks by roughly 7 microseconds per day. General relativity predicts that the same clocks, sitting higher in Earth's gravitational well where spacetime is less curved, run fast by roughly 45 microseconds per day. The effects act in opposite directions and do not cancel; the net result is that satellite clocks gain about 38 microseconds daily. Untreated, this would introduce a position error accumulating at about ten kilometres per day, rendering the system useless within hours. The correction is applied by offsetting the oscillator frequency before launch, so that the clocks run at the correct rate once in orbit, with residual relativistic terms from orbital eccentricity computed by the receiver.

The signal itself is a triumph of spread-spectrum design. Each satellite transmits on shared carrier frequencies using a distinct pseudorandom noise code, a technique called code division multiple access. The codes are constructed to have very low cross-correlation, so a receiver recovers one satellite's signal by correlating against its known code while treating all others as noise. Because the correlation gain is large, the signal can be transmitted at power levels that arrive at the surface below the thermal noise floor — roughly a hundred times weaker than background noise — and still be recovered. This is why GPS works with a small antenna and why it fails indoors, under dense canopy, and in urban canyons where buildings block or reflect the signal.

Accuracy is limited chiefly by the atmosphere. The ionosphere delays the signal by an amount that varies with electron content and, critically, with frequency, so dual-frequency receivers can measure and cancel the delay directly. The troposphere introduces a further delay that is not frequency-dependent and must be modelled. Augmentation systems broadcast correction data from reference stations at surveyed locations, improving accuracy from several metres to under a metre; real-time kinematic techniques, which track the carrier phase rather than the code, achieve centimetre precision and underpin precision agriculture and automated construction machinery.

Until May 2000 the system deliberately degraded civilian accuracy through Selective Availability, dithering the broadcast clock to produce errors around a hundred metres. Its removal by presidential directive improved civilian accuracy roughly tenfold overnight and catalysed the consumer navigation industry. Parallel constellations now operate: Russia's GLONASS, Europe's Galileo, and China's BeiDou, with modern receivers using several simultaneously for improved availability and integrity.

The system's ground segment is less visible than the constellation but equally essential. A master control station at Schriever Space Force Base in Colorado, supported by monitoring stations distributed around the globe, continuously tracks each satellite, measures the drift of its atomic clocks against a composite reference, and computes updated orbital elements. These corrections are uploaded at least daily and rebroadcast to users as the navigation message, since a satellite's orbit is perturbed by the oblateness of the Earth, by lunar and solar gravity, and by solar radiation pressure, to a degree that would otherwise degrade accuracy within hours. The broadcast ephemeris is accurate to roughly a metre; post-processed precise ephemerides, published by the International GNSS Service some days later, reach a few centimetres and are the basis of geodetic work such as measuring plate motion and monitoring ice sheet mass loss.

The dependence of critical infrastructure on a faint signal from space has become a recognised strategic vulnerability. Jamming, which simply drowns the signal in noise, requires only inexpensive hardware and is now routinely observed near conflict zones and around some airports, where illegal personal privacy devices in vehicles have repeatedly disrupted aviation systems. Spoofing is more insidious: a transmitter broadcasts counterfeit signals that a receiver accepts as genuine, and because civilian GPS signals are unencrypted and their structure is public, a receiver has no cryptographic means of authenticating them. Demonstrations have induced yachts and unmanned aircraft to accept false positions without any indication of anomaly, and large-scale spoofing affecting shipping in the Black Sea and the Persian Gulf has been documented. The military signal carries an encrypted code that resists both attacks, and Galileo has begun broadcasting an authenticated civilian service, but the enormous installed base of legacy receivers will remain vulnerable for decades.

A further concern is that GPS timing has become a single point of failure for systems whose operators often do not realise they depend on it. Cellular base stations require synchronisation to within microseconds to hand off calls; power grids use synchrophasor measurements to detect instability; and financial regulations in several jurisdictions mandate timestamp accuracy traceable to a national standard. Following a 2016 incident in which a decommissioning error introduced a thirteen-microsecond offset into the broadcast message, disrupting equipment worldwide for around eleven hours, several countries have invested in terrestrial backup timing, including enhanced Loran systems and fibre-based time distribution, on the reasoning that a utility this deeply embedded should not rest on a single technology.
"""},
    {"domain": "structured", "text": """
{"dataset": "customer_interactions", "version": "2.0", "schema": {"customer_id": "string", "interaction_timestamp": "timestamp", "interaction_type": "enum", "channel": "string", "sentiment": "float", "duration_seconds": "integer", "issue_resolved": "boolean", "agent_id": "string", "transcript_available": "boolean", "transcribed_text": "text_optional"}, "records": [{"customer_id": "C00123456", "interaction_timestamp": "2024-01-15T14:32:01Z", "interaction_type": "phone_support", "channel": "voice", "sentiment": 0.72, "duration_seconds": 847, "issue_resolved": true, "agent_id": "A0045", "transcript_available": true, "transcribed_text": "Customer called regarding billing discrepancy on monthly statement..."}, {"customer_id": "C00789012", "interaction_timestamp": "2024-01-15T15:01:15Z", "interaction_type": "chat_support", "channel": "messaging", "sentiment": 0.31, "duration_seconds": 1203, "issue_resolved": false, "agent_id": "A0078", "transcript_available": true, "transcribed_text": "Customer frustrated with product delivery delays..."}, {"customer_id": "C00456789", "interaction_timestamp": "2024-01-15T16:45:22Z", "interaction_type": "email_support", "channel": "email", "sentiment": 0.85, "duration_seconds": 2103, "issue_resolved": true, "agent_id": "A0091", "transcript_available": false, "transcribed_text": null}], "summary_statistics": {"total_records": 47832, "average_sentiment": 0.58, "issue_resolution_rate": 0.82, "average_duration_seconds": 1456, "date_range": {"start": "2024-01-01", "end": "2024-01-31"}}, "metadata": {"created_by": "analytics_team", "creation_date": "2024-02-01T08:15:00Z", "last_modified": "2024-02-05T10:30:00Z", "documentation_url": "https://wiki.internal.company/datasets/customer_interactions"}}
"""},
    {"domain": "structured", "text": """
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name example.com www.example.com;

    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/example.com/chain.pem;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' fonts.googleapis.com; font-src 'self' fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self' api.example.com;" always;

    root /var/www/html;
    index index.html index.htm;

    location / {
        try_files $uri $uri/ /index.html;
        expires -1;
        add_header Cache-Control "public, max-age=0";
    }

    location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|webp)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location ~* \\.well-known {
        allow all;
    }

    location ~ /\\. {
        deny all;
    }

    location ~ ~$ {
        deny all;
    }

    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }

    access_log /var/log/nginx/example.com.access.log combined buffer=32k;
    error_log /var/log/nginx/example.com.error.log warn;
}

server {
    listen 80;
    listen [::]:80;
    server_name example.com www.example.com;
    return 301 https://$server_name$request_uri;
}
"""},
    {"domain": "structured", "text": """
# Project Architecture and Setup

## Overview
This microservices application consists of five primary services communicating through message queues and REST APIs.

## Service Inventory

| Service Name | Language | Port | Database | Purpose |
|---|---|---|---|---|
| api-gateway | Node.js | 3000 | None | Request routing and auth |
| user-service | Python | 3001 | PostgreSQL | User management |
| inventory-service | Go | 3002 | Redis | Stock management |
| payment-service | Java | 3003 | PostgreSQL | Payment processing |
| notification-service | Python | 3004 | MongoDB | Email/SMS dispatch |

## Dependencies

### External Services
- PostgreSQL 13+ (users, payments)
- Redis 6+ (inventory cache)
- MongoDB 4+ (notifications)
- RabbitMQ 3.8+ (message queue)
- Auth0 (authentication)
- Stripe API (payments)
- SendGrid (email)
- Twilio (SMS)

### Internal Service Communication
```
Client -> API Gateway (port 3000)
    -> User Service (port 3001)
    -> Inventory Service (port 3002)
    -> Payment Service (port 3003)
    -> Notification Service (port 3004)
    -> Message Queue (RabbitMQ)
```

## Environment Configuration

### Required Variables
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `MONGODB_URI`: MongoDB connection string
- `RABBITMQ_URL`: RabbitMQ connection string
- `JWT_SECRET`: JWT signing key
- `STRIPE_API_KEY`: Stripe API credentials
- `SENDGRID_API_KEY`: SendGrid API key
- `TWILIO_ACCOUNT_SID`: Twilio account ID
- `TWILIO_AUTH_TOKEN`: Twilio authentication token

## Deployment

### Local Development
```bash
docker-compose up -d
npm run seed
npm run dev
```

### Production
Deployed via Kubernetes on AWS EKS with:
- Auto-scaling based on CPU and memory
- Service mesh for inter-service communication
- Persistent volumes for stateful services
- Ingress controller for external access
"""},
    {"domain": "structured", "text": """
LOG_DATE,LOG_TIME,SERVICE,LOG_LEVEL,MESSAGE,DURATION_MS,USER_ID,REQUEST_ID
2024-01-15,14:32:01.234,api-gateway,INFO,Request received: POST /api/v1/orders,0.5,U00123456,REQ-2024-01-15-001234
2024-01-15,14:32:01.456,user-service,INFO,User authenticated successfully,12.3,U00123456,REQ-2024-01-15-001234
2024-01-15,14:32:01.678,inventory-service,INFO,Checking stock availability,45.6,U00123456,REQ-2024-01-15-001234
2024-01-15,14:32:02.123,inventory-service,WARNING,Low stock warning: SKU-98765 (2 units remaining),78.9,U00123456,REQ-2024-01-15-001234
2024-01-15,14:32:02.456,payment-service,INFO,Payment processing initiated,156.7,U00123456,REQ-2024-01-15-001234
2024-01-15,14:32:03.012,payment-service,INFO,Payment authorized successfully,234.5,U00123456,REQ-2024-01-15-001234
2024-01-15,14:32:03.345,notification-service,INFO,Confirmation email queued,23.4,U00123456,REQ-2024-01-15-001234
2024-01-15,14:32:03.567,api-gateway,INFO,Order created successfully,1234.6,U00123456,REQ-2024-01-15-001234
2024-01-15,14:32:05.789,notification-service,INFO,Email sent successfully,456.7,U00123456,REQ-2024-01-15-001234
2024-01-15,14:33:12.345,user-service,ERROR,Database connection timeout after 5 retries,5000.0,U00789012,REQ-2024-01-15-001235
2024-01-15,14:33:12.678,api-gateway,ERROR,Request failed: 500 Internal Server Error,5234.2,U00789012,REQ-2024-01-15-001235
2024-01-15,14:35:01.234,inventory-service,INFO,Scheduled inventory sync started,0.1,SYSTEM,SCHEDULED-001
2024-01-15,14:35:15.456,inventory-service,INFO,Updated 1234 SKU records,14234.5,SYSTEM,SCHEDULED-001
2024-01-15,14:35:15.789,inventory-service,INFO,Scheduled inventory sync completed successfully,14234.8,SYSTEM,SCHEDULED-001
2024-01-15,14:36:45.123,payment-service,WARNING,Stripe API response slow: 2345ms response time,2345.0,U00456789,REQ-2024-01-15-001236
2024-01-15,14:36:45.456,payment-service,INFO,Payment completed despite slow API response,2567.8,U00456789,REQ-2024-01-15-001236
"""},
    {"domain": "dialogue", "text": """
"I don't think you're listening to me," Sarah said, setting down her coffee cup with more force than necessary. "This isn't about the money anymore. It's about respect. You promised we would talk about moving before making any decisions, and you just went ahead and signed the lease."

Mark sighed, rubbing his temples. "I know, and I'm sorry. I made a mistake. But the landlord needed an answer immediately, and the apartment was perfect. I thought you'd be excited about the space, the new neighborhood."

"That's exactly the problem," Sarah replied, her voice steadier now but edged with frustration. "You thought. You didn't ask. You didn't wait. After everything we've talked about, all those conversations about making decisions together, you still just went ahead."

"You're right," Mark said quietly. "You're completely right. I handled this badly. I got excited and I acted impulsively without considering that this affects both of us equally. What do you want to do?"

Sarah looked out the window, collecting herself. "I need to know that my opinion matters to you. That's what this is about. It's not the apartment. It's that you made a unilateral decision about our future without including me."

"I understand," Mark said. "And I want to make this right. What if we go look at the place together? If you hate it, we don't take it. I'll call the landlord today and tell them I need to verify with my partner before finalizing anything."

"You mean that?" Sarah asked, turning to face him.

"I do," Mark said. "I'm sorry it took me messing up to say this, but I should have been asking for your input from the beginning. Moving forward, I'm going to do better."

Sarah picked up her cup again, found it had gone cold, and set it down. "Okay. But I want to say the other part, because if I don't say it now I'll say it badly in three weeks."

"Go ahead."

"You do this when you're anxious," she said. "It's not a pattern of disrespect. It's a pattern of panic. Your mother called about the money in February and you reorganised our entire budget in one evening without telling me. The car, same thing. You get scared and you solve, and solving alone feels safer to you than deciding together."

Mark was quiet for a moment. "That's a fairly specific list."

"I've had time to think about it."

"No, it's—" He stopped. "It's accurate. I don't like that it's accurate. My father made every decision in that house and I've spent twenty years telling myself I'm nothing like him, and what you're describing is exactly him, just with a guilty conscience attached."

"I'm not calling you your father."

"I know. I'm calling me my father." He turned his mug a quarter turn on the table, a thing he did. "The apartment genuinely is good, by the way. That's what makes it worse. If it were a bad apartment this would be a simpler conversation."

"Tell me about it," Sarah said. "Not the pitch. The actual thing."

"Third floor, south-facing, so the front room gets light until about four. The kitchen is small. Smaller than ours. There's a second bedroom you could work in that doesn't share a wall with the bathroom." He paused. "The commute is worse for you by about fifteen minutes and I didn't factor that in until Tuesday."

"You didn't mention the commute."

"No."

"Why not?"

"Because I'd already signed," he said, "and I wanted it to be fine."

Sarah nodded slowly. "That's the answer I needed. Not the apology. That."

"What do you want to do?"

"I want to see it Saturday. And I want you to call the landlord tomorrow, on speaker, with me in the room, and find out honestly what the exit looks like. Not so I can catch you out. So that I know the real number instead of the reassuring one."

"That's fair," Mark said. "What if the real number is bad?"

"Then it's bad and we deal with it, and I'd rather deal with a bad number than a managed one." She looked at him. "That's the whole thing, Mark. I'm not fragile. You keep protecting me from information and calling it kindness."

He exhaled. "Saturday. And I'll call in the morning."

"Morning," she agreed. "And if I love it, I'm going to be insufferable about the fifteen minutes."

"I'd expect nothing less," he said, and she almost smiled, which was not forgiveness but was adjacent to it.
"""},
    {"domain": "dialogue", "text": """
"Tell me again why you think the company should invest in this," Dr. Patel said, leaning back in her chair, fingers steepled on the desk. "The projections look interesting, but the risk assessment concerns me."

James cleared his throat. "The market research is solid. We've identified a gap that competitors haven't addressed. Early adopters are ready, and our timeline gets us in before the market saturates."

"That's what you said about the last venture," Dr. Patel replied. "And we lost four million dollars when the market shifted faster than anticipated."

"Fair point," James acknowledged. "But this is different. We've incorporated feedback from that failure. We're building in flexibility for market changes, we've diversified our supplier base, and we're projecting more conservative growth numbers."

Dr. Patel nodded slowly. "I appreciate the honesty. What's your worst-case scenario?"

"We lose the initial investment," James said without hesitation. "We don't get sufficient market traction and the venture folds within eighteen months. Worst case, it damages our reputation in that sector."

"And best case?"

"We capture thirty percent market share within three years, the product becomes a standard in the industry, and we potentially sell the division for a significant multiple."

"The range is enormous," Dr. Patel said. "Why should I gamble billions on that spread?"

"Because the upside matches the risk," James said. "And because sometimes you have to take calculated risks to grow. But ultimately, this is your decision to make. If you're not comfortable, I understand. I can find another investor or restructure the proposal."

Dr. Patel smiled slightly. "That's exactly what I wanted to hear. You're not pushing too hard. You understand the stakes. I'll bring this to the board, but I'm leaning toward approval. If they ask questions, who should they direct them to?"

"Me, directly," James said. "I'll be available whenever they need."

Dr. Patel pulled the deck back toward her and turned to a page near the end. "Before I do that. Slide thirty-one. Walk me through the supplier concentration."

"Two vendors for the controller boards," James said. "Shenzhen and Penang."

"And if Penang goes offline."

"Shenzhen absorbs it at a forty percent unit cost increase and an eleven-week lead time."

"Which does what to your margin in year two?"

James didn't reach for the calculator, which she noted. "Takes it from thirty-one percent to about nineteen. We stay cash-positive but we lose the reinvestment headroom, so year three growth drops from the projection into the low teens."

"You've run that."

"I've run it, my CFO has run it independently, and we disagree by about two points on the tooling amortisation. Hers is the more conservative number and I've used hers in the deck."

Dr. Patel made a note. "That's the first thing you've said this morning that moved me."

"The disagreement?"

"That you used hers." She closed the folder. "Here's my difficulty, and I want you to hear it as a difficulty rather than an objection. The board approved the last venture on the strength of a presentation very like this one. Competent, honest about risk, well-modelled. And it failed. Not because anyone lied and not because anyone was lazy. Because a market moved in a direction nobody in this building had thought to model."

"I know," James said. "I was in the room."

"So what have you changed about how you think, rather than about what you're building? Because the second thing is in the deck and the first thing isn't."

James was quiet long enough that the silence became deliberate.

"I stopped trusting my own conviction as evidence," he said. "Last time I could feel that I was right, and that feeling was doing a lot of work in my reasoning without my noticing. This time I hired someone whose job includes telling me I'm wrong, and I gave her the model rather than the summary, and I've taken her numbers over mine twice. That's the change. It's not a very impressive answer."

"It's a considerably better answer than an impressive one," Dr. Patel said. She stood, which meant the meeting was ending. "I'll take it to the board Thursday. I'll recommend approval at seventy percent of what you asked for, structured in two tranches with the second gated on the year-one supplier diversification."

"Seventy percent doesn't get us to the launch window."

"No," she agreed. "It gets you to a launch window. You'll have to decide whether that's worth having. Think about it before Thursday and tell me honestly, because if you tell me yes and mean no, we'll have this conversation again in eighteen months with worse numbers."

James considered it. "I'll have an answer for you Wednesday."

"Wednesday is fine," she said. "And James — bring your CFO to the board meeting."
"""},
    {"domain": "dialogue", "text": """
"You've been avoiding me," Lisa said, not looking up from her book. "Three weeks of excuses about being busy."

From the kitchen doorway, her mother paused. "That's not fair. I've been genuinely busy with work."

"Mom," Lisa finally looked up, "I called you six times. SIX times. You answered once and we talked for three minutes before you said you had to go."

Her mother came and sat on the edge of the couch, choosing her words carefully. "I'm sorry. You're right. I haven't been present, and I know that hurts."

"Why?" Lisa asked. "What did I do? Are you upset with me about something?"

"No, honey, no," her mother said quickly. "This is about me, not you. I've been struggling since your father and I separated. I know that's not an excuse, but it's the reality. I've been trying to keep myself together and I've done that by isolating."

Lisa set her book aside. "I'm struggling too. I wanted to talk to you about school, about friends, about the divorce. But you weren't there."

"I know," her mother said, and her voice wavered. "I've been a terrible mother about this. I got so caught up in my own pain that I forgot you needed me. You're the child here, not me. I should have been taking care of you emotionally."

"I don't need perfect," Lisa said. "I just need you to try. To actually be present, not just physically here but actually listening to me."

"Starting now," her mother said, taking Lisa's hand. "I'm going to do better. I'm going to get some help for myself so I can be there for you. Will you help me? Will you tell me when I'm slipping back into old patterns?"

"Yeah," Lisa said, squeezing her hand. "I can do that."

They sat there for a while. The radiator made its noise.

"Can I ask you something," Lisa said, "and you don't do the voice."

"What voice?"

"The one where you answer like you're being recorded."

Her mother laughed, which turned into something else halfway through. "Okay. No voice."

"Is Dad coming back?"

"No," her mother said. "No, sweetheart. He isn't."

"Okay." Lisa pulled her sleeve down over her hand. "Everyone keeps doing this thing where they say it's complicated and adults are working it out, and I'm fifteen, I'm not six, and it makes me feel like I'm being handled."

"That's a completely fair thing to be angry about."

"I'm not angry, I just—" She stopped. "Okay, I'm a bit angry."

"You're allowed to be a lot angry."

"At you or at him?"

"At either. At both. It doesn't cost me anything," her mother said, "and you don't have to protect me. That's been the other thing I got wrong. I've watched you be careful around me for four months and I let you do it because it was easier."

Lisa looked at the carpet. "Aunt Jo said you weren't eating."

"Aunt Jo should mind her business," her mother said, and then, "she's right, though. I wasn't. I am now, mostly. That's a true answer and not a managing one."

"Mostly."

"Mostly. I'm seeing someone about it from the eighth. A proper one, through the GP, not a book."

Lisa nodded. Then, after a moment: "Can I say the bad thing?"

"Say the bad thing."

"Sometimes it's easier when you're not here. Because when you're here and you're like you've been, I have to think about you all the time. And when you were just gone I could just do my homework." She was crying now, in the annoyed way she had. "That's horrible. I don't mean it like it's—"

"It isn't horrible," her mother said. "It's the most useful thing you've said all year." She pulled a tissue from somewhere and handed it over. "That's exactly what it's like being a child around an adult who's struggling. And you shouldn't have had to learn it at fifteen."

"So what do we do?"

"Small things. I'm going to be here for dinner Thursdays and I'm going to ask you about school and I'm going to be capable of hearing the answer. That's the deal. If I stop being capable of hearing the answer, you say the word."

"What word?"

Her mother thought about it. "Thursday. Just say Thursday. I'll know."

Lisa laughed despite herself. "That's so stupid."

"It's extremely stupid," her mother agreed. "Are we doing it?"

"Yeah," Lisa said. "We're doing it."
"""},
    {"domain": "dialogue", "text": """
"I'm retiring," Thomas announced at the dinner table, watching carefully for his family's reaction. "Next month."

His daughter set down her fork. "Next month? Dad, we talked about this. We said you'd give us time to prepare, to understand the implications."

"I know what we said," Thomas replied. "But I'm sixty-eight. I don't have unlimited time. My health isn't getting better, and I want to enjoy retirement while I can actually enjoy it."

"That's selfish," his son said, then immediately held up his hands. "Sorry, that was harsh. But you're making a unilateral decision that affects all of us. Mom, your business, our plans."

His wife, Margaret, had been quiet. "Thomas, we talked about this. You promised you'd consult with me before making a final decision."

"I did consult," Thomas said. "We had conversations. But at some point, I have to make a decision about my own life."

"Your life doesn't exist in a vacuum," Margaret said, her voice steady but sad. "We're married. Your retirement affects my life, our finances, what we can and can't do together."

Thomas looked at his hands. "I'm scared," he said quietly. "I'm scared of getting sick, of losing time. Working is killing me slowly. I just want to have life back."

"Then let's figure this out together," Margaret said. "Don't just announce it at the dinner table. We can plan a retirement that works for both of us. But you have to include me in the process."

"I will," Thomas said. "I'm sorry. You're right. I was afraid you'd talk me out of it, so I just decided."

"I wouldn't talk you out of it," Margaret said. "But I need to be part of the planning. That's what marriage means."

Their daughter, Claire, put her fork down. "Can I say something as the person who actually has the spreadsheet?"

"You have a spreadsheet?" her brother said.

"I've had a spreadsheet since March. Dad asked me in March." She looked at her mother. "He did include someone. He just included the wrong someone, and I should have told him to talk to you, and I didn't, because I was flattered."

Margaret absorbed this. "March."

"March," Thomas said. "I'm not going to pretend that's better."

"It isn't better. It's worse, actually." Margaret's voice stayed level. "You've been carrying this for seven months and letting me plan a holiday for next autumn."

"Yes."

"Why?"

Thomas looked at his plate. "Because the first time I tried to say it you said 'don't be ridiculous, you love the work,' and I thought, she's right, and then I went and got the numbers from Claire anyway. And after that every day I didn't say it made saying it harder."

"I don't remember saying that."

"April. In the car, coming back from the Hendersons'."

Margaret was silent. "I do remember that," she said eventually. "I was thinking about the roof."

"I know. It wasn't a considered position. It just landed on me at the wrong angle."

Claire slid her phone across the table. "This is the actual picture. You go at sixty-eight, you draw the pension at seventy rather than bridging, and you're fine — genuinely fine, not fine-with-caveats — provided you don't do both the conservatory and the Portugal thing in the same three years."

Her son leaned over to look. "That's less bad than I'd assumed."

"It's less bad than Dad assumed too, which is partly why he's been so frightened about it."

Margaret took the phone, scrolled, and handed it back. "Right." She turned to Thomas. "Here's what I want. I want the whole thing in front of me by Sunday, including the bits you think will upset me. I want to talk to Dr. Aziz with you at the next appointment, because I think the health part of this is doing more work in your reasoning than you've admitted at this table."

Thomas nodded slowly.

"And then," Margaret said, "assuming all of that, I think you should retire. Probably in March rather than next month, because next month is a tantrum and March is a decision. But yes."

Thomas looked up. "You think I should?"

"Thomas, I've been watching you come home grey for two years." She picked up her glass. "I wasn't going to be the one to say it, because I thought you'd hear it as being pushed. Which is, I suppose, exactly the same mistake in the opposite direction."

"March," Thomas said, testing it.

"March," Margaret agreed. "And you can tell them yourself. I'm not doing that part for you."
"""},
    {"domain": "dialogue", "text": """
"The test results came back," the doctor said, sliding the folder across the desk. "I wish I had better news."

Elena felt her chest tighten. "Just tell me straight. What are we looking at?"

"Cancer, stage two," Dr. Morrison said quietly. "In the lymph nodes. It's treatable, but treatment will be aggressive. We're talking chemotherapy, possibly radiation, likely surgery."

"How long?" Elena asked. "Do I have... how much time are we talking?"

"That's the question we can't answer with certainty," Dr. Morrison said. "Stage two with aggressive treatment gives us reasonable hope. I'd say five-year survival rates are around sixty to seventy percent, but that's just statistics. Each case is different."

"My daughter is getting married in eight months," Elena said. "Will I be able to walk her down the aisle?"

Dr. Morrison leaned forward. "Possibly. If you begin treatment soon, eight months is enough time for at least initial treatment cycles, and depending on how you respond, you might be in a manageable place by then. But I won't lie to you—this will be difficult."

Elena felt tears starting. "I don't want my daughter to remember me sick."

"Then we talk about when and how you're comfortable telling her," Dr. Morrison said. "We develop a treatment plan. We manage side effects. And we focus on quality of life, not just quantity. Some patients surprise themselves with what they're capable of doing even during treatment."

"What's my next step?" Elena asked, wiping her eyes.

"We consult with an oncologist this week," Dr. Morrison said. "We design a treatment protocol tailored to your specific cancer. And we talk about support—counseling, support groups, family involvement. You don't do this alone."

"I want to walk my daughter down the aisle," Elena said firmly.

"Then let's make sure we do everything we can to make that happen," Dr. Morrison replied.

Elena took a breath. "Okay. Practical questions. Can I write them down?"

"Please do. Take as long as you need."

"Will I lose my hair?"

"With the regimen I expect Dr. Whitcomb will propose, yes, almost certainly, starting around week three. It grows back. I mention the timing because patients tell me the not knowing when is worse than the thing itself."

"Week three." Elena wrote it down. "Can I work?"

"Some people can, some can't, and it's genuinely not predictable in advance. What I'd suggest is that you talk to your employer about intermittent leave now, while you're not yet in it, rather than negotiating from a bad week in April. Have you got someone in HR who's reasonable?"

"Marisol. She's good."

"Talk to Marisol."

Elena wrote that down too. "The wedding is the twelfth of September. If treatment starts when?"

"If you see the oncologist Thursday, I'd expect a start date within two to three weeks. So call it early February. That gives you roughly six cycles before September with a gap at the end."

"So I could be finished."

"You could be finished with the chemotherapy portion, yes. Whether you're finished with surgery and radiation depends on what the imaging shows after cycle three." Dr. Morrison paused. "Elena, I want to say one thing carefully. You've now asked me three questions about September. That's completely natural. But I've watched people organise an entire year around a single date and then feel they've failed if the date goes badly, and I don't want that for you."

Elena's jaw tightened. "It's my daughter's wedding."

"I know. And I'm not telling you to let go of it." He leaned back. "I'm telling you that Sofia would rather have you at the reception in a chair than have you exhaust yourself standing. Has she said anything about it?"

"She doesn't know yet."

"When are you telling her?"

"I don't know. After the dress fitting, maybe. She's been so happy."

"That's your decision and there's no wrong answer," Dr. Morrison said. "But I'd offer this: the longer the gap between your knowing and her knowing, the more likely she is to feel the gap itself as the injury. Not the diagnosis. The being kept outside of it."

Elena stared at the folder for a long moment. "My mother did that to me. She had a stroke in 2004 and I found out from a neighbour."

"How did that sit?"

"Badly. It sat very badly." She closed the notebook. "Alright. This weekend. I'll tell her this weekend."

"That sounds right." Dr. Morrison stood and walked her to the door. "Thursday, ten fifteen, third floor. Bring someone with you — not for support, though that's fine too, but because you will not retain half of what's said and a second set of ears is clinically useful."

"I'll bring Sofia," Elena said, and found that saying it steadied her.
"""},
    {"domain": "structured", "text": """
package_name,version,maintainer,license,dependencies,installation_date,update_available,annual_downloads,github_stars,security_vulnerabilities,last_security_audit
numpy,1.24.3,Travis E. Oliphant,BSD,Python>=3.9,2024-01-10,1.25.0,100000000,28500,0,2024-01-08
pandas,2.0.2,"Wes McKinney, others",BSD,"numpy>=1.21.6, python>=3.9",2024-01-15,2.0.3,85000000,41800,1,2024-01-10
scikit-learn,1.2.2,"Gael Varoquaux, others",BSD,"numpy, scipy, joblib",2024-01-08,1.3.0,45000000,58000,0,2024-01-09
matplotlib,3.7.1,"John D. Hunter, others",PSF,"python>=3.9, numpy, pillow",2024-01-09,3.7.2,65000000,18000,2,2024-01-05
requests,2.31.0,Kenneth Reitz,Apache 2.0,"charset-normalizer, idna, urllib3, certifi",2024-01-11,none,120000000,51000,1,2024-01-12
django,4.2.0,"Django Software Foundation",BSD,"Python>=3.8, asgiref, sqlparse, tzdata",2024-01-07,4.2.1,28000000,72000,3,2024-01-11
flask,2.3.2,Armin Ronacher,BSD,"Python>=3.8, Werkzeug, Jinja2, click, itsdangerous",2024-01-12,2.3.3,35000000,65000,0,2024-01-09
pytorch,2.0.0,"Facebook AI, contributors",BSD,"Python>=3.8, numpy",2024-01-05,none,22000000,72000,2,2024-01-13
tensorflow,2.12.0,"Google Brain Team",Apache 2.0,"Python>=3.8, numpy, protobuf, h5py",2024-01-03,2.13.0,18000000,180000,4,2024-01-14
jupyter,1.0.0,"The Jupyter Development Team",BSD,"notebook, jupyter_console, jupyter_client",2024-01-14,none,15000000,10000,0,2024-01-06
scipy,1.10.1,"Travis E. Oliphant, others",BSD,"numpy>=1.21.6, python>=3.9",2024-01-09,1.11.0,70000000,12000,1,2024-01-07
"""},
    {"domain": "structured", "text": """
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>microservices-app</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>

    <name>Microservices Application</name>
    <description>A distributed microservices architecture</description>

    <properties>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <spring.boot.version>3.0.0</spring.boot.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <version>${spring.boot.version}</version>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
            <version>${spring.boot.version}</version>
        </dependency>
        <dependency>
            <groupId>org.postgresql</groupId>
            <artifactId>postgresql</artifactId>
            <version>42.5.0</version>
            <scope>runtime</scope>
        </dependency>
        <dependency>
            <groupId>io.jsonwebtoken</groupId>
            <artifactId>jjwt-api</artifactId>
            <version>0.12.3</version>
        </dependency>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <version>${spring.boot.version}</version>
            </plugin>
        </plugins>
    </build>
</project>
"""},
    {"domain": "structured", "text": """
# Kubernetes Deployment Configuration

---
apiVersion: v1
kind: Namespace
metadata:
  name: production

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
        version: v1.2.0
    spec:
      containers:
      - name: api-server
        image: company/api-server:1.2.0
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: api-server-service
  namespace: production
spec:
  selector:
    app: api-server
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""},
    {"domain": "structured", "text": """
# Application Environment Configuration File
# This file contains all environment variables required for the application
# DO NOT commit this file to version control - use .env.example instead
# Generated: 2024-01-15T10:00:00Z

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/app_db
DATABASE_POOL_SIZE=20
DATABASE_TIMEOUT=30000
DATABASE_REPLICA_URL=postgresql://user:password@localhost:5432/app_db_replica
DATABASE_ENABLE_LOGGING=true

# Redis Cache Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_CLUSTER_NODES=redis-1:6379,redis-2:6379,redis-3:6379
REDIS_PASSWORD=your-redis-password-here
REDIS_MAX_RETRIES=3
REDIS_RETRY_DELAY=1000
REDIS_TTL_SECONDS=3600

# JWT and Authentication
JWT_SECRET=your-secret-key-here-minimum-32-characters-for-security
JWT_EXPIRY_HOURS=24
JWT_REFRESH_SECRET=your-refresh-secret-key-minimum-32-characters
JWT_REFRESH_EXPIRY_DAYS=7
OAUTH_PROVIDER=auth0
OAUTH_CLIENT_ID=your-oauth-client-id
OAUTH_CLIENT_SECRET=your-oauth-client-secret

# Server Configuration
NODE_ENV=development
LOG_LEVEL=debug
API_PORT=3000
API_HOST=localhost
CORS_ORIGIN=http://localhost:3000,http://localhost:3001
CORS_CREDENTIALS=true
CORS_MAX_AGE=86400

# Payment Gateway Configuration
STRIPE_API_KEY=sk_test_xxxxxxxxxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxx
STRIPE_API_VERSION=2023-10-16
PAYMENT_TIMEOUT_SECONDS=30

# Email Configuration
SENDGRID_API_KEY=SG.xxxxxxxxxxxx
SENDGRID_FROM_EMAIL=noreply@company.com
SENDGRID_FROM_NAME=Company Notifications
EMAIL_TEMPLATE_ID=d-xxxxxxxxxxxx

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_S3_BUCKET=company-bucket
AWS_S3_REGION=us-east-1
AWS_CLOUDFRONT_DOMAIN=d111111abcdef8.cloudfront.net

# Error Tracking and Monitoring
SENTRY_DSN=https://xxxx@sentry.io/xxxxxxx
SENTRY_ENVIRONMENT=development
SENTRY_SAMPLE_RATE=0.1
DATADOG_API_KEY=xxxxxxxxxxxx
DATADOG_APP_KEY=xxxxxxxxxxxx

# Rate Limiting Configuration
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_BY_IP=true
RATE_LIMIT_BY_USER=true

# Session Configuration
SESSION_TIMEOUT_MINUTES=30
SESSION_COOKIE_NAME=app_session
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=strict

# Cache Configuration
CACHE_TTL_SECONDS=3600
CACHE_BACKEND=redis
CACHE_KEY_PREFIX=app_

# Feature Flags
ENABLE_ANALYTICS=true
ANALYTICS_SAMPLE_RATE=0.1
ENABLE_NEW_DASHBOARD=false
ENABLE_BETA_API=false
ENABLE_EXPERIMENTAL_FEATURES=false

# Third-party Services
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1234567890
"""},
    {"domain": "dialogue", "text": """
"I need to talk about something difficult," Rachel said, setting her coffee down and looking directly at her brother. "About how you've been treating Mom."

Jonathan's jaw tightened. "Here we go again. What did I do this time?"

"You canceled on her twice this month," Rachel said quietly. "She's struggling with Dad's memory care issues, and she needs support. Your support specifically."

"I'm busy," Jonathan said defensively. "Work has been insane. I'm doing the best I can."

"I understand work is busy," Rachel replied. "But so is caregiving, and she's doing that alone because you've checked out."

Jonathan stood up. "That's not fair. I help financially. I've been sending money for her assisted living situation."

"Money isn't the same as presence," Rachel said. "She doesn't need your money. She needs her son. She needs you to show up."

"And what about you?" Jonathan asked angrily. "You're calling me out, but you live twenty minutes away and you only visit once a week."

"You're right," Rachel said, surprising him. "I'm not doing enough either. But at least I'm trying to expand that. You're going backward. And I'm calling you out because I love you and I don't want you to wake up five years from now and realize you missed the chance to be present for her."

Jonathan sat back down slowly. "I didn't realize it was affecting her that much."

"She doesn't complain to you," Rachel said. "She just says she understands you're busy. But I see it. She lights up when you call, and then she's disappointed when you cancel."

"I don't want to be the person who abandons family," Jonathan said quietly. "But I'm also barely keeping my head above water."

"Then let's figure out a realistic schedule," Rachel said. "Even once a month is better than nothing. Or maybe you call her weekly. Something consistent that you can actually commit to."

"One Sunday a month," Jonathan said. "Dinner or something. I can do that."

"She'd love that," Rachel said.

Jonathan turned his coffee cup around. "Can I ask you something without you treating it as a deflection?"

"Depends what it is."

"What's the actual medical situation? Because what I get from Mom is 'your father's doing fine, he had a good week,' and what I get from you is that it's bad, and I can't reconcile those two things from four hundred miles away."

Rachel looked at him. "That's a fair question and I've been assuming you didn't want to know."

"I didn't want to know in about 2021," Jonathan said. "That's a different thing from now."

"Okay." She pushed her plate aside. "He doesn't know who she is about half the time. He's pleasant about it — he's never been aggressive, some of them get aggressive — but he treats her like a very nice member of staff. Last month he thanked her for coming and asked when his wife was getting back."

Jonathan didn't say anything.

"The good weeks are real," Rachel said. "They're just shorter and further apart, and she reports them to you because they're the part she can bear to say out loud. She's not lying to you. She's rationing."

"Jesus."

"The practical thing is the nights. He's up at two, three, sometimes four, and he wants to go to work. Fully dressed, looking for his keys. And she can't sleep through it because he'll get out the front door, so she's been on about four hours a night since roughly October."

"Since October." Jonathan set the cup down. "She told me in November that she'd never slept better."

"Yes. She would."

He was quiet for a while. "One Sunday a month isn't the right offer, is it."

"I wasn't going to say that."

"You were about to construct an environment in which I said it."

Rachel almost smiled. "Slightly."

"What's actually useful? Not what's fair, what's useful. Because I have money and I have about two functional weekends a quarter, and I'd rather spend both on the right thing than spread them thin and feel good about it."

"Respite care," Rachel said immediately. "Three nights a week, overnight, so she sleeps. It's about four hundred a week and she won't spend it because she thinks it's giving up. She'd take it from you. She wouldn't take it from me — we've had that fight twice."

"Why would she take it from me?"

"Because you're the son and she's never been able to refuse you anything," Rachel said, without evident bitterness. "It's irritating and it's also a resource, and I'd like to use it."

Jonathan nodded slowly. "I'll call her tonight. And I'll come the last weekend of February, and I'll do the nights while I'm there so you can stop."

"I don't do the nights."

"I know," Jonathan said. "You should probably start being honest with me about that part too."
"""},
    {"domain": "dialogue", "text": """
"We should buy a house," Marcus said, watching his partner's face carefully. "In the next year or two."

Casey looked up from their laptop. "A house? In this market? Have you seen the prices?"

"I know," Marcus said. "Which is why we should start saving now. We're throwing away money on rent, and interest rates are only going up."

"But we're not ready," Casey said. "We haven't finished paying off your student loans. We don't have the down payment saved. What if we lose our jobs?"

"Those are excuses," Marcus said, and he heard the tension in his own voice. "We could be ready if we prioritized it. We could cut back on dining out, travel—"

"Wait," Casey interrupted. "So now it's my fault? Because I don't want to eliminate everything fun from our lives?"

"That's not what I said," Marcus replied. "I'm saying we have to make sacrifices to achieve goals. Financial stability matters."

"It does matter," Casey said. "But you're not asking me what I want. You're telling me what we should do and expecting me to fall in line."

Marcus felt his defensiveness rising. "Is homeownership not something you want?"

"Eventually, maybe," Casey said. "But on our timeline, not yours. I like where we are. I like having flexibility. I'm not ready to lock ourselves into a mortgage right now."

Marcus was quiet for a long moment. "So you're saying no."

"I'm saying let's talk about it more," Casey said. "Let's understand what both of us actually want, not just what sounds responsible. And let's make decisions together, not with one person trying to convince the other that their timeline is correct."

"You're right," Marcus said slowly. "I was pushing my anxiety onto you. I got scared about money and I wanted to control the situation."

"I appreciate that acknowledgment," Casey said. "And I'm willing to think about saving more aggressively. But I need to feel like I have a voice in this decision."

"You do," Marcus said. "And I'm listening."

Casey closed the laptop. "Then let me say the part I've been sitting on, because I don't think it's actually about houses."

"Okay."

"When you say financial stability, I hear your mother's voice. And I know that's a heavy thing to say at nine on a Tuesday, but you say it in her cadence, and it comes up every time work is uncertain for you."

Marcus started to answer and stopped.

"Was there a restructure?" Casey asked.

"There's a consultation process," Marcus said. "It was announced Thursday. Eleven roles in my group, they're keeping eight."

"Thursday."

"Thursday."

"And tonight you opened with mortgages."

"Yes."

Casey rubbed their face. "Marcus."

"I know."

"No, I don't think you do, so let me be precise about why this is a problem. If you'd come home Thursday and said 'I'm frightened, there's a consultation, I need us to look at our runway' — I would have been on that spreadsheet with you inside an hour. I'm good at that. That's genuinely a thing I'm good at." Casey's voice had risen slightly. "Instead you spent four days converting fear into a plan and then presented the plan, and when I pushed back on the plan you heard it as me being reckless with our future. And I couldn't argue properly because I was arguing with the wrong thing."

"That's fair," Marcus said quietly.

"Are the eight decided?"

"Not formally. I think I'm probably in the eight. My manager's been unusually friendly, which is either very good or very bad."

"What's the timeline?"

"Provisional outcomes end of March. Notice periods after that."

Casey pulled the laptop back and opened a new document. "Right. Then here's what we do, and none of it involves a house. We work out what we spend in a month — the real number, not the aspirational one. We work out how long savings last if you're in the three, including the redundancy payment, which you haven't mentioned and which is probably substantial after six years. We find out what your notice period actually is. And then, when we know the real shape of it, we decide together whether we're a household that's frightened or a household that's inconvenienced."

Marcus watched them type. "And if we're frightened?"

"Then we're frightened with numbers, which is enormously better than frightened without them." Casey looked up. "And I'll say the other thing now so you don't have to ask. If it goes badly, I can cover us both for about eight months on my salary if we drop the travel. I did that sum in November, because I do that sum every November."

"You never told me that."

"You never told me you were scared," Casey said. "Shall we call it even and start over from the actual problem?"

Marcus laughed, once, unsteadily. "Yeah. Let's do that."
"""},
    {"domain": "dialogue", "text": """
"The numbers don't make sense," the financial advisor said, sliding the spreadsheet across the table. "If you maintain current spending, you'll run out of money by seventy-five."

Patricia felt her stomach drop. "But I thought my pension and social security—"

"They help significantly," the advisor interrupted, not unkindly. "But they're not sufficient given your lifestyle. We need to make changes."

"Like what?" Patricia asked, already dreading the answer.

"Move to a less expensive area," the advisor said. "Downsize your home. Reduce discretionary spending. You might need to work longer or find part-time work in retirement."

Patricia had been planning to retire in two years. The thought of working longer filled her with despair. "Is there another way?"

"Not without significant sacrifice," the advisor said. "Unless there's family money coming, or you're willing to accept living quite a bit more modestly."

"My sister might leave me something," Patricia said quietly. "But I can't count on that."

"You can't," the advisor agreed. "We work with what we know for certain."

Patricia looked at the numbers. Numbers that represented her future, her security, her ability to have the life she'd envisioned. "I need some time to think about this."

"I understand," the advisor said. "But I'd recommend starting to make changes soon. The earlier you adjust, the less dramatic the changes need to be."

Patricia nodded, feeling overwhelmed. "What would happen if I moved somewhere cheaper? Like, significantly cheaper?"

"Your money would stretch much further," the advisor said. "Your housing cost is your biggest expense. If you cut that in half, the numbers look very different."

"Would I have to leave my city?" Patricia asked. "My friends, my community—"

"Not necessarily," the advisor said. "But you might need to move to a smaller place within the city, or look at towns nearby."

Patricia left the office feeling shaken. All her life she'd been responsible, careful with money. And yet here she was, facing the possibility that she'd have to completely reimagine her retirement.

She went back three weeks later with a legal pad and a different posture.

"I've done some arithmetic of my own," she said, sitting down. "And I have questions about your assumptions, because I think two of them are wrong."

The advisor — his name was Dennis, and she had decided to start using it — raised his eyebrows. "Go on."

"You've got my spending flat in real terms from sixty-seven to ninety-five. That's not how people spend. Everything I've read says it falls in the seventies and eighties and then rises at the end if there's care."

"That's correct," Dennis said. "The literature calls it the retirement smile. I model flat because it's conservative."

"Conservative for whom? It's conservative for you, because if I run out of money at eighty-eight nobody blames you for being too gloomy." She kept her voice pleasant. "It's not conservative for me, because it's currently telling me to sell a house I don't want to sell on the strength of an assumption you've just agreed is unrealistic."

Dennis was quiet for a moment. "That's a legitimate criticism. Can I re-run it with a declining real spend and a care shock at eighty-five?"

"Please. And the second thing. You've got the house as a fixed cost. It isn't. It's a four-bedroom with two empty rooms and a lodger rate in that postcode of about eleven hundred a month."

"You'd take a lodger?"

"I had a lodger for nine years in the nineties," Patricia said. "I know exactly what it involves, which is more than most people who say they'd consider it. A postgraduate, term-time, through the university's accredited scheme. Rent-a-room relief means the first seven and a half thousand is untaxed."

Dennis pulled the spreadsheet toward him and began typing. "If that's nine thousand net a year from sixty-seven to, say, seventy-eight..."

"And the declining spend."

"And the declining spend." He worked for a couple of minutes. Patricia watched the number at the bottom of the screen and did not let herself hope at it.

"You don't run out," Dennis said eventually. "On these assumptions you die at ninety-five with about eighty thousand left, which is roughly a year of residential care, which is thin but it isn't destitution."

"On these assumptions."

"On these assumptions. And I want to be honest that I've just moved two levers you asked me to move, which is not the same as the situation having improved."

"No," Patricia agreed. "But it's the difference between a plan and a sentence, and I've spent three weeks thinking I'd been handed a sentence." She capped her pen. "What's the thing I should actually be frightened of? Not the thing that makes the worst chart. The thing you'd tell a friend."

Dennis considered it properly. "Care costs," he said. "Everything else is manageable. If you need residential care for more than about two years, no arrangement we can make today survives it, and that's true of nearly everyone I see."

"Then let's talk about that," Patricia said, "instead of the house."
"""},
]

import json
import pathlib
import sys

DEFAULT_OUT = pathlib.Path("corpus512.jsonl")


def main(argv: list[str]) -> int:
    out = pathlib.Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, sample in enumerate(SAMPLES):
            record = {"id": str(i), "domain": sample["domain"], "text": sample["text"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(SAMPLES)} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
