import { pgTable, text, boolean, timestamp, doublePrecision, jsonb } from 'drizzle-orm/pg-core';

export const users = pgTable('users', {
  id: text('id').primaryKey(),
  username: text('username').notNull().unique(),
  firstName: text('first_name'),
  email: text('email').notNull().unique(),
  hashedPassword: text('hashed_password').notNull(),
  isActive: boolean('is_active').default(true),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const analysisJobs = pgTable('analysis_jobs', {
  id: text('id').primaryKey(),
  modality: text('modality').notNull(),
  status: text('status').default('PENDING').notNull(),
  message: text('message'),
  fileName: text('file_name'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
  userId: text('user_id').references(() => users.id),
});

export const forensicResults = pgTable('forensic_results', {
  id: text('id').primaryKey(),
  jobId: text('job_id').references(() => analysisJobs.id).notNull().unique(),
  verdict: text('verdict').notNull(),
  fakeProbability: doublePrecision('fake_probability').notNull(),
  authenticityScore: doublePrecision('authenticity_score').notNull(),
  confidence: doublePrecision('confidence').notNull(),
  rawResults: jsonb('raw_results'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const passwordResets = pgTable('password_resets', {
  token: text('token').primaryKey(),
  userId: text('user_id').references(() => users.id).notNull(),
  expiresAt: timestamp('expires_at').notNull(),
});
